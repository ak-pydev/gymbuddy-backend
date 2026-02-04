import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
import torch
import numpy as np

def test_ntu120_loader():
    print("Initializing NTU120Dataset...")
    dataset = NTU120Dataset(data_path='data/raw/skeleton/ntu120/ntu120_3d.pkl', target_frames=60)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Fetch one sample
    idx = 0
    sample = dataset[idx]
    
    x = sample['x']
    y = sample['y']
    mask = sample['mask']
    
    print(f"Sample {idx}:")
    print(f"  x shape: {x.shape} (Expected: (60, 25, 3))")
    print(f"  y: {y} (Type: {type(y)})")
    print(f"  mask shape: {mask.shape}")
    
    assert x.shape == (60, 25, 3), f"x shape mismatch: {x.shape}"
    assert isinstance(y, int) or isinstance(y, np.integer), f"y is not int: {type(y)}"
    
    # Check normalization
    # Joint 0 should be 0,0,0 at every frame (since we subtracted it)
    # Allow small float error
    pelvis = x[:, 0, :]
    max_pelvis_val = torch.max(torch.abs(pelvis))
    print(f"  Max absolute value at Joint 0 (should be ~0): {max_pelvis_val}")
    
    assert max_pelvis_val < 1e-5, "Normalization (centering) failed."
    
    print("Test passed!")

if __name__ == "__main__":
    test_ntu120_loader()
