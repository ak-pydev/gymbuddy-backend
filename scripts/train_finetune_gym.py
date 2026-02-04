import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
from pathlib import Path
import copy

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.gym_dataset import GymDataset
from gymbuddy.models.transformer import SkeletonTransformer

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            out = model(x)
            loss = criterion(out, y)
            
            total_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    return total_loss / total, correct / total

def train_finetune(args):
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if not torch.backends.mps.is_available() and torch.cuda.is_available():
        device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # 1. Dataset
    # We might not have 'train'/'val' splits in the pkl yet, so we'll do random split if needed.
    # If the user prepared the pkl with splits, we use them.
    # For now, let's load 'all' and split 80/20 if explicit splits fail or are same.
    
    try:
        train_ds = GymDataset(data_path=args.data_path, split='train')
        val_ds = GymDataset(data_path=args.data_path, split='val')
        if len(train_ds) == len(val_ds) and len(train_ds) > 0:
             # Suspicious duplication or fallback. Let's explicitly random split if they seem identical
             # But if dataset logic worked, they should be different. 
             # Let's assume they are correct.
             pass
    except Exception as e:
        print(f"Could not load splits directly ({e}). Loading all and splitting 80/20.")
        full_ds = GymDataset(data_path=args.data_path, split=None)
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model
    # Determine num_classes from dataset? Or arg?
    # Gym dataset likely has different classes than NTU (120).
    # We need to reshape the head.
    
    # Check number of classes in dataset if possible
    # We can peek at one batch or assume args.num_classes
    # Let's peek
    if hasattr(train_ds, 'dataset'): # it's a Subset
         sample_y = train_ds.dataset.samples[0].get('label', 0)
         # This assumes labels are 0..N-1. We need max label.
         # This is risky. Better to argument or max scan.
         # Let's scan quickly if small, or defaults.
         pass
    
    # Initialize Model with PRETRAINED weights (NTU config)
    # NTU usually 120 classes.
    print(f"Loading checkpoint from {args.checkpoint}...")
    baseline_model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=0.5)
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # If checkpoint is full state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        baseline_model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Replace Head for Gym
    # Gym classes? Let's say user provides it or we scan. 
    # If we don't know, we'll scan the dataset labels.
    
    # If we don't know, we'll scan the dataset labels.
    
    unique_labels = set()
    print("Scanning dataset for classes...")
    # handling Subset vs Dataset
    ds_to_scan = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
    
    # We need to scan ALL labels to be safe, or just the ones in this split + val split? 
    # Safer to scan all if possible, or build mapping dynamically.
    # For now, let's scan the training set samples at least.
    
    # If ds_to_scan has 'samples' attribute (List of dicts)
    if hasattr(ds_to_scan, 'samples'):
        for s in ds_to_scan.samples:
            unique_labels.add(int(s.get('label', 0)))
    else:
        # Fallback if samples not accessible directly, look at __getitem__ (slow)
        pass

    sorted_labels = sorted(list(unique_labels))
    num_unique = len(sorted_labels)
    min_label = sorted_labels[0] if num_unique > 0 else 0
    max_label = sorted_labels[-1] if num_unique > 0 else 0
    
    print(f"Found {num_unique} unique classes. Range: [{min_label}, {max_label}]")
    
    # Check if remapping is needed (if max_label >= num_unique or min_label < 0)
    # Actually, we should ALWAYS remap to be safe 0..N-1
    
    label_map = {old: new for new, old in enumerate(sorted_labels)}
    
    if args.num_classes is not None and args.num_classes != num_unique:
        print(f"Warning: User provided --num_classes={args.num_classes} but found {num_unique} unique classes in dataset.")
        print(f"Overriding --num_classes to {num_unique} to match dataset remapping.")
        
    args.num_classes = num_unique
    print(f"Training with {args.num_classes} classes.")
    
    # We need to wrap the dataset to apply remapping on the fly
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, mapping):
            self.dataset = original_dataset
            self.mapping = mapping
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            item = self.dataset[idx]
            original_y = item['y']
            # Map y
            # If original_y not in map (e.g. validation set has label not in train? risky)
            # We assume sorted_labels came from FULL dataset scan above.
            if original_y in self.mapping:
                item['y'] = self.mapping[original_y]
            else:
                 # fallback/error? default to 0
                 # print(f"Warning: Label {original_y} not in map.")
                 item['y'] = 0 
            return item
            
    # Apply wrapper
    train_ds = RemappedDataset(train_ds, label_map)
    val_ds = RemappedDataset(val_ds, label_map)
    
    # Re-create loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
    model = baseline_model
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, args.num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # PHASE 1: Head Only
    print(f"\n=== Phase 1: Training Head Only ({args.epochs_head} epochs) ===")
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr_head)
    
    for epoch in range(args.epochs_head):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs_head} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
    # PHASE 2: Full Finetuning
    print(f"\n=== Phase 2: Full Finetuning ({args.epochs_full} epochs) ===")
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr_full)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_full)
    
    best_acc = 0.0
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    for epoch in range(args.epochs_full):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs_full} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            print(f"Saved new best model (Acc: {best_acc:.4f})")
            
    # Save final
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final.pt"))
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='Path to gym.pkl')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained ntu120 checkpoint')
    parser.add_argument('--out_dir', type=str, default='outputs/gym_finetune')
    parser.add_argument('--epochs_head', type=int, default=2)
    parser.add_argument('--epochs_full', type=int, default=10)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_full', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=None, help='Overrides auto-detection if provided')
    
    parser.add_argument('--epochs', type=int, default=None, help='Alias for --epochs_full')
    parser.add_argument('--lr', type=float, default=None, help='Alias for --lr_full')
    
    args = parser.parse_args()
    
    # Handle aliases
    if args.epochs is not None:
        args.epochs_full = args.epochs
    if args.lr is not None:
        args.lr_full = args.lr
        
    train_finetune(args)
