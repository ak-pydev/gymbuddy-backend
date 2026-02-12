import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from gymbuddy.data.loaders.ntu120 import NTU120Dataset

def analyze_per_class(mc_file, data_path, out_dir, dataset_name="Gym"):
    print(f"Loading MC outputs from {mc_file}...")
    data = np.load(mc_file)
    probs = data['probs'] # [N, n_passes, n_classes]
    labels = data['labels']
    
    # Average probs over MC passes
    mean_probs = probs.mean(axis=1)
    preds = mean_probs.argmax(axis=1)
    
    n_classes = mean_probs.shape[1]
    
    # Compute per-class accuracy
    class_accs = []
    class_counts = []
    
    for c in range(n_classes):
        mask = (labels == c)
        count = mask.sum()
        class_counts.append(count)
        if count > 0:
            acc = (preds[mask] == labels[mask]).mean()
            class_accs.append(acc)
        else:
            class_accs.append(0.0) # Or NaN
            
    class_accs = np.array(class_accs)
    
    # Get class names
    # Try to load from dataset or use dummy
    try:
        # Hacky way to get class names if we don't want to instantiate the whole dataset
        # typically hardcoded or in a file
        class_names = [f"Class {i}" for i in range(n_classes)] 
        # If we had the label map... 
        # For NTU120, we can try to load it.
        # But let's check if the dataset class has a list.
        # Assuming NTU120Dataset has a class_names attribute or similar.
        # For now, we will stick to indices if not found.
        pass
    except:
        pass

    # Sort by accuracy
    sorted_indices = np.argsort(class_accs)
    
    # Top 5 and Bottom 5 (considering only classes with samples)
    valid_indices = [i for i in sorted_indices if class_counts[i] > 0]
    
    top_5 = valid_indices[-5:][::-1]
    bottom_5 = valid_indices[:5]
    
    print("\nTop 5 Classes:")
    for i in top_5:
        print(f"  Class {i}: {class_accs[i]:.4f} ({class_counts[i]} samples)")
        
    print("\nBottom 5 Classes:")
    for i in bottom_5:
        print(f"  Class {i}: {class_accs[i]:.4f} ({class_counts[i]} samples)")
        
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Combine top and bottom for the plot
    plot_indices = top_5 + bottom_5
    plot_accs = [class_accs[i] for i in plot_indices]
    plot_labels = [f"Class {i}" for i in plot_indices]
    colors = ['g']*5 + ['r']*5
    
    y_pos = np.arange(len(plot_indices))
    ax.barh(y_pos, plot_accs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Accuracy')
    ax.set_title(f'{dataset_name}: Best & Worst Performing Classes')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{dataset_name}_per_class_transfer.png'))
    plt.close()
    print(f"Saved plot to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mc_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default="Gym")
    args = parser.parse_args()
    
    out_dir = os.path.join(args.out_dir, 'training_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    analyze_per_class(args.mc_file, args.data_path, out_dir, args.dataset_name)
