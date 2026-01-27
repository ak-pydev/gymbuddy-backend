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
        low = bin_boundaries[i]
        high = bin_boundaries[i+1]
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
            conf = confidences[mask].mean()
            ece += np.abs(acc - conf) * mask.sum() / total
            accs.append(acc)
            confs.append(conf)
            counts.append(mask.sum())
            
    return ece, (accs, confs, counts)

def make_plots():
    parser = argparse.ArgumentParser(description='Generate calibration plots')
    parser.add_argument('--mc_file', type=str, required=True, help='Path to MC dropout .npz output')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save figures')
    args = parser.parse_args()

    results_path = args.mc_file
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return
    
    print(f"Loading {results_path}...")
    data = np.load(results_path)
    probs = data['p_mean']
    uncertainty = data['u_epistemic']
    labels = data['y_true']
    
    # ECE
    ece, (accs, confs, counts) = compute_ece_bins(probs, labels)
    print(f"MC Dropout ECE: {ece:.4f}")
    
    with open(os.path.join(output_dir, "ece.json"), "w") as f:
        json.dump({"ece": ece}, f)

    # Reliability Plot
    plt.figure()
    plt.plot([0,1], [0,1], "k--")
    plt.plot(confs, accs, "o-")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.4f})")
    plt.savefig(os.path.join(output_dir, "reliability.png"))
    plt.close()
    
    # Uncertainty vs Error
    preds = np.argmax(probs, axis=1)
    errors = (preds != labels).astype(int)
    
    plt.figure()
    # Boxplot of uncertainty for Correct vs Wrong
    u_correct = uncertainty[errors == 0]
    u_wrong = uncertainty[errors == 1]
    
    plt.boxplot([u_correct, u_wrong], labels=['Correct', 'Wrong'])
    plt.ylabel("Epistemic Uncertainty")
    plt.title("Uncertainty vs Prediction Error")
    plt.savefig(os.path.join(output_dir, "uncertainty_error.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    make_plots()
