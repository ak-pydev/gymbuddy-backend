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

def eval_mc():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint_path = "outputs/ntu120_baseline/checkpoint.pt"
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found. Run training first.")
        # Create dummy for code check if needed, but here we assume flow.
        return

    # Load Config (Assume default or load json)
    # Using hardcoded roughly matching train script
    model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=0.5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # Dataset (Val)
    val_ds = NTU120Dataset(split='xsub_val', target_frames=60)
    # For quick eval, maybe subset? But prompt says "paper-ready evaluation".
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print("Running MC Dropout Prediction...")
    probs, uncertainty, labels = predict_mc(model, val_loader, n_passes=10, device=device)
    
    # 1. ECE
    ece, (accs, confs, counts) = compute_ece(probs, labels)
    print(f"MC Dropout ECE: {ece:.4f}")
    
    # 2. Plots
    os.makedirs("outputs/figs", exist_ok=True)
    
    # Reliability Diagram
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.scatter(confs, accs, s=np.array(counts)/np.sum(counts)*1000, alpha=0.5) # Size proportional to count
    plt.plot(confs, accs, "r-", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.legend()
    plt.savefig("outputs/figs/reliability_diagram.png")
    plt.close()
    
    # Uncertainty vs Error
    # Bin samples by uncertainty
    preds = np.argmax(probs, axis=1)
    errors = (preds != labels).astype(int)
    
    # Sort by uncertainty
    sorted_indices = np.argsort(uncertainty)
    uncertainty_sorted = uncertainty[sorted_indices]
    errors_sorted = errors[sorted_indices]
    
    # Sliding window or bins
    n_bins_u = 10
    u_bins = np.array_split(uncertainty_sorted, n_bins_u)
    e_bins = np.array_split(errors_sorted, n_bins_u)
    
    mean_u = [np.mean(b) for b in u_bins]
    mean_e = [np.mean(b) for b in e_bins]
    
    plt.figure()
    plt.plot(mean_u, mean_e, "o-")
    plt.xlabel("Predictive Uncertainty (Variance)")
    plt.ylabel("Error Rate")
    plt.title("Uncertainty vs Error")
    plt.grid(True)
    plt.savefig("outputs/figs/uncertainty_vs_error.png")
    plt.close()
    
    # Save raw results
    np.savez("outputs/ntu120_baseline/mc_results.npz", probs=probs, uncertainty=uncertainty, labels=labels)
    print("Evaluation Complete. Results saved.")

if __name__ == "__main__":
    eval_mc()
