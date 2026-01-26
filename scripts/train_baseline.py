import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
from gymbuddy.models.baseline import SimpleSkeletonModel

def train_baseline():
    # Settings
    SUBSET_SIZE = 2000 # Use 2k samples for sanity check
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-3
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if not torch.backends.mps.is_available() and torch.cuda.is_available():
        device = torch.device('cuda')
        
    print(f"Using device: {device}")
    
    # Dataset
    dataset = NTU120Dataset(data_path='data/raw/skeleton/ntu120/ntu120_3d.pkl', target_frames=60)
    
    # Subset
    indices = np.arange(len(dataset))
    # Shuffle indices
    np.random.shuffle(indices)
    subset_indices = indices[:SUBSET_SIZE]
    
    subset = Subset(dataset, subset_indices)
    
    # Split Train/Val
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_ds, val_ds = random_split(subset, [train_size, val_size])
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = SimpleSkeletonModel(num_classes=120)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            
            # Acc
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        avg_loss = total_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                out = model(x)
                
                _, predicted = torch.max(out.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Final check
    if val_acc > 1/120:
        print("Success: Validation accuracy > random chance (0.0083)")
    else:
        print("Warning: Validation accuracy is low. Check model or data.")

if __name__ == "__main__":
    train_baseline()
