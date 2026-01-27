import sys
import os
import torch
import numpy as np
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset

def debug_data():
    splits = ['xsub_train', 'xsub_val']
    
    label_sets = {}
    
    for split in splits:
        print(f"\nChecking split: {split}")
        ds = NTU120Dataset(split=split, target_frames=60)
        print(f"Total samples: {len(ds)}")
        
        # Collect all labels (might take a moment)
        # To be fast, sample 2000 random
        indices = np.random.choice(len(ds), min(5000, len(ds)), replace=False)
        labels = [ds.samples[i]['label'] for i in indices]
        
        unique = set(labels)
        label_sets[split] = unique
        print(f"Unique labels in sample: {len(unique)}")
        print(f"Min: {min(unique)}, Max: {max(unique)}")
        
    # Check overlap
    train_labels = label_sets['xsub_train']
    val_labels = label_sets['xsub_val']
    
    print("\nOverlap Analysis:")
    print(f"Union: {len(train_labels | val_labels)}")
    print(f"Intersection: {len(train_labels & val_labels)}")
    print(f"In Train only: {len(train_labels - val_labels)}")
    print(f"In Val only: {len(val_labels - train_labels)}")
    
    if len(train_labels | val_labels) != 120:
        print("WARNING: Total unique labels found is not 120 (might be sampling issue or NTU60 subset?).")

if __name__ == "__main__":
    debug_data()
