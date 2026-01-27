import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import random

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
from gymbuddy.models.transformer import SkeletonTransformer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import argparse

def train_full():
    parser = argparse.ArgumentParser(description='Train Skeleton Transformer on NTU120')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4) # Reduced from 1e-3
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--split', type=str, default='xsub', help='xsub or xview')
    parser.add_argument('--debug', action='store_true', help='Run on a small subset for verification')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    args = parser.parse_args()
    
    # Configuration
    CONFIG = vars(args)
    CONFIG['device'] = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(CONFIG['seed'])
    # Updated output dir
    output_dir = "outputs/ntu120_xsub_baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
    # Save yaml manually to avoid pyyaml dependency if missing
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        for k, v in CONFIG.items():
            f.write(f"{k}: {v}\n")
        
    print(f"Starting training with config: {CONFIG}")
    
    # Datasets
    print("Initializing Datasets...")
    train_ds = NTU120Dataset(split=f"{CONFIG['split']}_train", target_frames=60)
    val_ds = NTU120Dataset(split=f"{CONFIG['split']}_val", target_frames=60)
    
    if CONFIG['debug']:
        print("DEBUG MODE: Using subset of data (100 samples)")
        indices = np.random.choice(len(train_ds), 100, replace=False)
        train_ds = torch.utils.data.Subset(train_ds, indices)
        
        v_indices = np.random.choice(len(val_ds), 20, replace=False)
        val_ds = torch.utils.data.Subset(val_ds, v_indices)
        
        CONFIG['epochs'] = 2
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Model
    model = SkeletonTransformer(
        num_classes=120,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'] if 'nhead' in CONFIG else 4,
        num_layers=CONFIG['num_layers'] if 'num_layers' in CONFIG else 4,
        dropout=CONFIG['dropout']
    )
    model.to(CONFIG['device'])
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
    
    # Scheduler with Warmup
    # Standard: linear warmup for 5% of epochs, then cosine decay.
    # We can use SequentialLR.
    total_steps = CONFIG['epochs']
    warmup_steps = int(0.05 * total_steps)
    if warmup_steps < 1: warmup_steps = 1
    
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
    
    # Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0.0
    metrics = {'train_loss': [], 'val_acc': [], 'lr': []}
    
    # helper for csv
    csv_file = os.path.join(output_dir, "train_curve.csv")
    with open(csv_file, "w") as f:
        f.write("epoch,train_loss,val_acc,lr\n")

    for epoch in range(CONFIG['epochs']):
        # ... (training loop same as before) ...
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        current_lr = scheduler.get_last_lr()[0]
        
        for batch in train_loader:
            x = batch['x'].to(CONFIG['device'])
            y = batch['y'].to(CONFIG['device']).long()
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        avg_loss = total_loss / (total + 1e-6)
        train_acc = correct / (total + 1e-6)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(CONFIG['device'])
                y = batch['y'].to(CONFIG['device']).long()
                out = model(x)
                _, predicted = torch.max(out.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_acc = val_correct / (val_total + 1e-6)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        metrics['train_loss'].append(avg_loss)
        metrics['val_acc'].append(val_acc)
        metrics['lr'].append(current_lr)
        
        # Append to CSV
        with open(csv_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{val_acc:.4f},{current_lr:.8f}\n")
        
        scheduler.step()
        
        # Checkpoint (Save best)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            print(f"  New Best Val Acc: {best_val_acc:.4f}. Saving best.pt...")
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))
            
        # Save last
        torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))
            
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    print("Training Complete.")

if __name__ == "__main__":
    train_full()
