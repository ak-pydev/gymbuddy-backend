#!/usr/bin/env python3
"""
Fine-tune NTU-pretrained model on gym_2d data.

Supports two fine-tuning strategies:
1. Head-only: Freeze backbone, train classification head only
2. Full fine-tune: Train all layers with lower learning rate

This enables Option B experiments: what improves with minimal adaptation.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import json
from scipy import interpolate
from tqdm import tqdm
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from gymbuddy.models.transformer import SkeletonTransformer


def interpolate_frames(skeleton, target_frames):
    """Interpolate skeleton to target frames using linear interpolation."""
    original_frames, num_joints, num_coords = skeleton.shape
    
    if original_frames == target_frames:
        return skeleton
    
    original_indices = np.linspace(0, 1, original_frames)
    target_indices = np.linspace(0, 1, target_frames)
    
    interpolated = np.zeros((target_frames, num_joints, num_coords))
    
    for j in range(num_joints):
        for c in range(num_coords):
            f = interpolate.interp1d(original_indices, skeleton[:, j, c], kind='linear')
            interpolated[:, j, c] = f(target_indices)
    
    return interpolated


def normalize_skeleton(skeleton, center_joint=0):
    """Normalize skeleton: center on joint, scale to unit bbox."""
    T, J, C = skeleton.shape
    center = skeleton[:, center_joint:center_joint+1, :]
    centered = skeleton - center
    max_val = np.abs(centered).max()
    if max_val > 1e-6:
        centered = centered / max_val
    return centered


def load_gym_data(data_path, target_frames=60, normalize=True, center_joint=0):
    """Load and preprocess gym data."""
    print(f"Loading gym data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        if 'skeletons' in data:
            skeletons, labels = data['skeletons'], data['labels']
        elif 'x' in data:
            skeletons, labels = data['x'], data['y']
        elif 'annotations' in data:
            # Format: {'split': ..., 'annotations': [{'keypoint': array, 'label': int}, ...]}
            annotations = data['annotations']
            print(f"  Found annotations format with {len(annotations)} samples")
            skeletons = [ann['keypoint'] for ann in annotations]
            labels = [ann['label'] for ann in annotations]
        else:
            raise ValueError(f"Unknown dict keys: {data.keys()}")
    elif isinstance(data, tuple):
        skeletons, labels = data
    elif isinstance(data, list):
        # List of (skeleton, label) tuples or list of dicts
        if isinstance(data[0], dict):
            skeletons = [item['keypoint'] for item in data]
            labels = [item['label'] for item in data]
        else:
            skeletons = [item[0] for item in data]
            labels = [item[1] for item in data]
    else:
        raise ValueError(f"Unknown format: {type(data)}")
    
    # Keep as list for now (variable length sequences)
    labels = np.array(labels)
    
    print(f"  Original: {len(skeletons)} samples, first shape: {np.array(skeletons[0]).shape}")
    
    # Preprocess each skeleton individually
    processed = []
    for skel in skeletons:
        skel = np.array(skel)
        
        # Handle extra dimension if present
        while skel.ndim > 3:
            skel = skel.squeeze(0)
        
        # If shape is (M, T, J, C) for M persons, take first
        if skel.ndim == 4:
            skel = skel[0]
        
        if skel.shape[0] != target_frames:
            skel = interpolate_frames(skel, target_frames)
        if normalize:
            skel = normalize_skeleton(skel, center_joint)
        processed.append(skel)
    
    skeletons = np.array(processed)
    print(f"  Processed: shape {skeletons.shape}")
    
    return skeletons, labels


def create_dataloaders(skeletons, labels, batch_size=32, val_split=0.2):
    """Create train/val dataloaders."""
    X = torch.tensor(skeletons, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    train_ds, val_ds = random_split(dataset, [n_train, n_val], 
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    return train_loader, val_loader


def freeze_backbone(model):
    """Freeze all layers except classification head."""
    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name and 'head' not in name:
            param.requires_grad = False
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in tqdm(loader, desc='Train', leave=False):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in tqdm(loader, desc='Val', leave=False):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * len(y)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    
    return total_loss / total, correct / total


def finetune_gym():
    parser = argparse.ArgumentParser(description='Fine-tune on gym_2d')
    parser.add_argument('--gym_data', type=str, required=True, help='Path to gym_2d.pkl')
    parser.add_argument('--checkpoint', type=str, required=True, help='NTU-pretrained checkpoint')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_classes', type=int, default=120, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--mode', type=str, choices=['head', 'full'], default='head',
                        help='head=freeze backbone, full=train all layers')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--target_frames', type=int, default=60, help='Target frames')
    parser.add_argument('--no_normalize', action='store_true', help='Skip normalization')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {args.checkpoint}...")
    model = SkeletonTransformer(
        num_classes=args.num_classes,
        d_model=256, nhead=4, num_layers=4, dropout=0.5
    )
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    # Freeze if head-only
    if args.mode == 'head':
        print(f"\nMode: HEAD-ONLY fine-tuning")
        freeze_backbone(model)
    else:
        print(f"\nMode: FULL fine-tuning (all layers)")
    
    model.to(device)
    
    # Load data
    print("\nLoading gym data...")
    skeletons, labels = load_gym_data(
        args.gym_data,
        target_frames=args.target_frames,
        normalize=not args.no_normalize
    )
    
    train_loader, val_loader = create_dataloaders(
        skeletons, labels, 
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10}")
    print("-" * 60)
    
    best_val_acc = 0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        print(f"{epoch+1:>6} {train_loss:>12.4f} {train_acc:>10.4f} {val_loss:>12.4f} {val_acc:>10.4f}")
        
        # Save best and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'mode': args.mode
            }, os.path.join(args.out_dir, 'best.pt'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break
        
        scheduler.step()
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'val_acc': val_acc,
        'mode': args.mode
    }, os.path.join(args.out_dir, 'last.pt'))
    
    # Save history
    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    config['final_val_acc'] = val_acc
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Mode: {args.mode}")
    print(f"  Best Val Acc: {best_val_acc:.4f}")
    print(f"  Saved to: {args.out_dir}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print(f"  1. Evaluate with: python scripts/gym_2d/eval_gym.py \\")
    print(f"       --gym_data {args.gym_data} \\")
    print(f"       --checkpoint {args.out_dir}/best.pt \\")
    print(f"       --out_file outputs/uncertainty/gym_finetuned_mc.npz")


if __name__ == "__main__":
    finetune_gym()
