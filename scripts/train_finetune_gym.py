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
    
    # 1. Load FULL Dataset first to scan labels globally
    print(f"Loading full dataset from {args.data_path or 'default'}...")
    full_ds = GymDataset(data_path=args.data_path, split=None)
    
    # 2. Scan classes from FULL dataset
    unique_labels = set()
    print("Scanning full dataset for classes...")
    
    # If full_ds has 'samples' attribute (List of dicts)
    if hasattr(full_ds, 'samples'):
        for s in full_ds.samples:
            # Handle potential Tensor vs int
            lbl = s.get('label', 0)
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
            unique_labels.add(int(lbl))
    else:
        # Fallback 
        for i in range(len(full_ds)):
             item = full_ds[i] 
             lbl = item['y']
             if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
             unique_labels.add(int(lbl))

    sorted_labels = sorted(list(unique_labels))
    num_unique = len(sorted_labels)
    min_label = sorted_labels[0] if num_unique > 0 else 0
    max_label = sorted_labels[-1] if num_unique > 0 else 0
    
    print(f"Found {num_unique} unique classes. Range: [{min_label}, {max_label}]")
    print(f"Original Labels: {sorted_labels}")

    # 3. Create Mapping
    label_map = {old: new for new, old in enumerate(sorted_labels)}
    
    # Override num_classes logic
    if args.num_classes is not None and args.num_classes != num_unique:
        print(f"Warning: User provided --num_classes={args.num_classes} but found {num_unique} unique classes in dataset.")
        print(f"Overriding --num_classes to {num_unique} to match dataset remapping.")
        
    args.num_classes = num_unique
    print(f"Training with {args.num_classes} classes (remapped 0..{args.num_classes-1}).")
    
    # 4. Wrap Dataset
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, mapping):
            self.dataset = original_dataset
            self.mapping = mapping
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            item = self.dataset[idx]
            original_y = item['y']
            
            # Robust conversion
            if isinstance(original_y, torch.Tensor):
                original_y = original_y.item()
            original_y = int(original_y)
            
            if original_y in self.mapping:
                new_y = self.mapping[original_y]
            else:
                 # Should not happen if we scanned full dataset
                 new_y = 0 
            
            # Return as Long Tensor
            item['y'] = torch.tensor(new_y, dtype=torch.long)
            return item

    # Apply wrapper to FULL dataset
    full_ds_remapped = RemappedDataset(full_ds, label_map)
    
    # 5. Split Train/Val
    # Check if original had split info?
    # For now, we just do random split on the remapped full dataset
    train_size = int(0.8 * len(full_ds_remapped))
    val_size = len(full_ds_remapped) - train_size
    train_ds, val_ds = random_split(full_ds_remapped, [train_size, val_size])
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    # Print stats of remapped
    # (Optional: could scan train_ds to show class distribution)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
    model = baseline_model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=args.dropout)
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
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    
    parser.add_argument('--epochs', type=int, default=None, help='Alias for --epochs_full')
    parser.add_argument('--lr', type=float, default=None, help='Alias for --lr_full')
    
    args = parser.parse_args()
    
    # Handle aliases
    if args.epochs is not None:
        args.epochs_full = args.epochs
    if args.lr is not None:
        args.lr_full = args.lr
        
    train_finetune(args)
