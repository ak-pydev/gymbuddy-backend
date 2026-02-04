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

def train_one_epoch(model, loader, criterion, optimizer, device, debug_batch=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, batch in enumerate(loader):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        # DEBUG: Check input shape and labels once
        if debug_batch and i == 0:
            print(f"DEBUG Batch 0: x shape={x.shape}, y shape={y.shape}")
            print(f"DEBUG Batch 0: y type={y.dtype}, y contents={y[:8].tolist()}")
            if x.dim() != 4:
                 print(f"ERROR: Expected 4D input (B, T, J, C), got {x.dim()}D {x.shape}")
            # Ensure indices valid
            print(f"DEBUG Batch 0: y min={y.min()}, max={y.max()}")
        
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # DEBUG: Inspect Datasets
    def inspect(split_name, ds, n=2000):
        # Handle subset
        if isinstance(ds, torch.utils.data.Subset):
           indices = ds.indices[:n]
           ys = [int(ds.dataset[i]['y'].item()) for i in indices]
        else:
           ys = [int(ds[i]['y'].item()) for i in range(min(len(ds), n))]
           
        u = sorted(set(ys))
        print(f"DEBUG {split_name}: n={len(ds)} sampled={len(ys)} unique={len(u)} min={min(u) if u else 'N/A'} max={max(u) if u else 'N/A'} first10={u[:10]}")
    
    inspect("train", train_ds)
    inspect("val", val_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # DEBUG: Overfit tiny subset
    if args.debug_overfit:
        print("!!! DEBUG MODE: Overfitting a tiny subset (64 samples) !!!")
        tiny_ds, _ = random_split(train_ds, [64, len(train_ds)-64])
        train_loader = DataLoader(tiny_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(tiny_ds, batch_size=32, shuffle=False) # validate on train
        

        
    # 2. Initialize Model & Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = baseline_model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=args.dropout)
    
    # Load state dict
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        
        # Load params (strict=False in case of minor mismatches, though usually should match)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"Missing keys (example): {missing[:5]}")
            
    except Exception as e:
        print(f"CRITICAL ERROR loading checkpoint: {e}")
        return

    # 3. Replace Head
    in_features = model.fc.in_features
    print(f"Replacing head: {in_features} -> {args.num_classes}")
    
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
    
    # DEBUG: Check trainable params
    trainable = [(n, p.shape) for n,p in model.named_parameters() if p.requires_grad]
    print(f"DEBUG: Trainable params (Head Only): {len(trainable)}")
    if len(trainable) > 0:
        print(f"DEBUG: Example trainables: {trainable[:5]}")
    
    for epoch in range(args.epochs_head):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs_head} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
    # PHASE 2: Full Finetuning
    print(f"\n=== Phase 2: Full Finetuning ({args.epochs_full} epochs) ===")
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
        
    # DEBUG: Check trainable params again
    trainable = [(n, p.shape) for n,p in model.named_parameters() if p.requires_grad]
    print(f"DEBUG: Trainable params (Full): {len(trainable)}")
    
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
    
    if args.debug_overfit:
        print("DEBUG: Tiny subset training complete. Did accuracy reach ~1.0?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='Path to gym.pkl')
    parser.add_argument('--checkpoint', type=str, default='/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs/ntu120_xsub_baseline/best.pt', help='Path to pretrained ntu120 checkpoint')
    parser.add_argument('--out_dir', type=str, default='outputs/gym_finetune')
    parser.add_argument('--epochs_head', type=int, default=2)
    parser.add_argument('--epochs_full', type=int, default=10)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_full', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=None, help='Overrides auto-detection if provided')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--debug_overfit', action='store_true', help='Sanity check: overfit small subset')
    
    parser.add_argument('--epochs', type=int, default=None, help='Alias for --epochs_full')
    parser.add_argument('--lr', type=float, default=None, help='Alias for --lr_full')
    
    args = parser.parse_args()
    
    # Handle aliases
    if args.epochs is not None:
        args.epochs_full = args.epochs
    if args.lr is not None:
        args.lr_full = args.lr
        
    train_finetune(args)
