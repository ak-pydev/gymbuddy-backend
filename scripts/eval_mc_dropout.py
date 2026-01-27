import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
from gymbuddy.models.transformer import SkeletonTransformer
from gymbuddy.uncertainty.mc_dropout import predict_mc

def compute_ece(probs, labels, n_bins=10):
    """ Expected Calibration Error """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accs = []
    confs = []
    counts = []
    
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    ece = 0.0
    total_samples = len(labels)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Indices in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            acc_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()
            
            ece += np.abs(acc_in_bin - conf_in_bin) * np.sum(in_bin) / total_samples
            
            accs.append(acc_in_bin)
            confs.append(conf_in_bin)
            counts.append(np.sum(in_bin))
            
    return ece, (accs, confs, counts)

def eval_mc(debug=False):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Path to baseline checkpoint
    checkpoint_path = "outputs/ntu120_xsub_baseline/best.pt"
    # Output path for uncertainty stats
    output_dir = "outputs/uncertainty"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ntu120_xsub_mc.npz")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Run training first.")
        # Try fallback
        old_path = "outputs/ntu120_xsub_baseline/checkpoint.pt"
        if os.path.exists(old_path):
             print(f"Found checkpoint at old path name {old_path}, utilizing.")
             checkpoint_path = old_path
        else:
             return

    # Load Model (Ensure d_model match)
    model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=0.5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # Dataset (Val)
    print("Initializing Validation Dataset...")
    val_ds = NTU120Dataset(split='xsub_val', target_frames=60)
    
    if debug:
        print("DEBUG MODE: Using subset of 50 samples")
        indices = np.random.choice(len(val_ds), 50, replace=False)
        val_ds = torch.utils.data.Subset(val_ds, indices)
        
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print(f"Running MC Dropout Prediction (N=20)...")
    probs, uncertainty, labels = predict_mc(model, val_loader, n_passes=20, device=device)
    
    # Compute predictions (y_pred) from mean probabilities
    y_pred = np.argmax(probs, axis=1)
    
    # Save raw results
    print(f"Saving results to {output_file}...")
    np.savez(output_file, 
             p_mean=probs, 
             u_epistemic=uncertainty, 
             y_true=labels, 
             y_pred=y_pred)
    print("Inference Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run on a small subset')
    args = parser.parse_args()
    eval_mc(debug=args.debug)
