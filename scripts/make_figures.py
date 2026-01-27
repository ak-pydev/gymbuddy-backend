import numpy as np
import matplotlib.pyplot as plt
import os
import json

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
        
        if np.sum(in_bin) > 0:
            acc_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()
            
            ece += np.abs(acc_in_bin - conf_in_bin) * np.sum(in_bin) / total_samples
            
            accs.append(acc_in_bin)
            confs.append(conf_in_bin)
            counts.append(np.sum(in_bin))
            
    return ece, (accs, confs, counts)

def make_plots():
    input_file = "outputs/uncertainty/ntu120_xsub_mc.npz"
    output_dir = "outputs/figs"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
        
    print(f"Loading {input_file}...")
    data = np.load(input_file)
    # New keys: p_mean, u_epistemic, y_true
    probs = data['p_mean']
    uncertainty = data['u_epistemic']
    labels = data['y_true']
    
    # 1. ECE
    ece, (accs, confs, counts) = compute_ece(probs, labels)
    print(f"MC Dropout ECE: {ece:.4f}")
    
    # Save ECE
    with open(os.path.join(output_dir, "ece.json"), "w") as f:
        json.dump({"ece": ece}, f, indent=4)
        
    # 2. Reliability Diagram
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(confs, accs, "r-", marker='o', label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reliability.png"))
    plt.close()
    
    # 3. Uncertainty vs Error
    preds = np.argmax(probs, axis=1)
    errors = (preds != labels).astype(int)
    
    sorted_indices = np.argsort(uncertainty)
    uncertainty_sorted = uncertainty[sorted_indices]
    errors_sorted = errors[sorted_indices]
    
    # Binning
    n_bins_u = 15
    u_bins = np.array_split(uncertainty_sorted, n_bins_u)
    e_bins = np.array_split(errors_sorted, n_bins_u)
    
    mean_u = [np.mean(b) for b in u_bins]
    mean_e = [np.mean(b) for b in e_bins]
    
    plt.figure(figsize=(6, 6))
    plt.plot(mean_u, mean_e, "o-", linewidth=2)
    plt.xlabel("Predictive Uncertainty (Variance)")
    plt.ylabel("Error Rate")
    plt.title("Uncertainty vs Error Rate")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "uncertainty_vs_error.png"))
    plt.close()
    
    print("Plots saved to outputs/figs/")

if __name__ == "__main__":
    make_plots()
