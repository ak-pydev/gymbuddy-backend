import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.kinetics_skeleton import KineticsSkeletonDataset

def test_kinetics_loader():
    print("Testing Kinetics Loader...")
    
    # We use 'temp' as data_root where we extracted one file
    # We need to find the index in annotations that corresponds to 'W-qaXgFW70Y'
    dataset = KineticsSkeletonDataset(
        meta_path='data/raw/skeleton/kinetics400/k400_2d.pkl',
        data_root='temp',
        target_frames=60,
        split='train' # Assume it's in train, or we iterate all
    )
    
    target_id = 'W-qaXgFW70Y'
    found_idx = -1
    
    for i in range(len(dataset)):
        # We access internal samples list to avoid loading file yet
        if dataset.samples[i]['frame_dir'] == target_id:
            found_idx = i
            break
            
    if found_idx == -1:
        # Check val split
        print(f"{target_id} not found in train. Checking val...")
        dataset_val = KineticsSkeletonDataset(
            meta_path='data/raw/skeleton/kinetics400/k400_2d.pkl',
            data_root='temp',
            target_frames=60,
            split='val'
        )
        for i in range(len(dataset_val)):
            if dataset_val.samples[i]['frame_dir'] == target_id:
                found_idx = i
                dataset = dataset_val
                break
                
    if found_idx != -1:
        print(f"Found {target_id} at index {found_idx}")
        sample = dataset[found_idx]
        
        x = sample['x']
        conf = sample['confidence']
        print(f"x shape: {x.shape} (Expected (60, 17, 2))")
        print(f"confidence shape: {conf.shape} (Expected (60, 17))")
        
        assert x.shape == (60, 17, 2)
        assert conf.shape == (60, 17)
        # Check if non-zero (since we have the file)
        if torch.sum(x) == 0:
            print("Warning: x is all zeros. File load failed?")
        else:
            print("Data loaded successfully (non-zero).")
            
    else:
        print(f"Sample {target_id} not found in metadata. Cannot test loading.")

if __name__ == "__main__":
    test_kinetics_loader()
