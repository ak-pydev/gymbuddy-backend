import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
from gymbuddy.models.transformer import SkeletonTransformer

def run_overfit():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load small subset
    ds = NTU120Dataset(split='xsub_train', target_frames=60)
    indices = np.arange(32) # Fixed 32 samples
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    # Model
    model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=0.0) # No dropout for overfit
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Overfit Test (Target: Acc -> 1.0, Loss -> 0.0)")
    
    model.train()
    for step in range(500):
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device).long()
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            # checks
            _, preds = torch.max(out, 1)
            acc = (preds == y).float().mean().item()
            
            if step % 50 == 0:
                print(f"Step {step}: Loss {loss.item():.4f}, Acc {acc:.4f}")
                
            if acc == 1.0 and loss.item() < 0.01:
                print("SUCCESS: Overfitted!")
                return

    print("FAILED: Did not perfectly overfit in 500 steps.")

if __name__ == "__main__":
    run_overfit()
