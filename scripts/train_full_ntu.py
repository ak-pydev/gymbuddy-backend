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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--split', type=str, default='xsub', help='xsub or xview')
    parser.add_argument('--debug', action='store_true', help='Run on a small subset for verification')
    args = parser.parse_args()
    
    # Configuration
    CONFIG = vars(args)
    CONFIG['device'] = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(CONFIG['seed'])
    output_dir = "outputs/ntu120_baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
        
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
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0) # 0 workers for debug safety
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Model (rest same as before)
    model = SkeletonTransformer(
        num_classes=120,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'] if 'nhead' in CONFIG else 4,
        num_layers=CONFIG['num_layers'] if 'num_layers' in CONFIG else 4,
        dropout=CONFIG['dropout'] if 'dropout' in CONFIG else 0.5
    )
    model.to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    best_val_acc = 0.0
    metrics = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
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
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        metrics['train_loss'].append(avg_loss)
        metrics['val_acc'].append(val_acc)
        scheduler.step()
        
        # Checkpoint
        if val_acc >= best_val_acc: # >= to save initial
            best_val_acc = val_acc
            print(f"  New Best Val Acc: {best_val_acc:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint.pt"))
            
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    print("Training Complete.")

if __name__ == "__main__":
    train_full()
