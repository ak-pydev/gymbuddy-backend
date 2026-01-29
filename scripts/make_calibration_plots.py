import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import sys
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None
from sklearn.metrics import confusion_matrix

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
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            
            accs.append(acc_in_bin)
            confs.append(conf_in_bin)
            counts.append(np.sum(in_bin))
            
    return ece, (accs, confs, counts)

def plot_reliability_diagram(accs, confs, ece, out_path):
    plt.figure(figsize=(6, 6))
    plt.plot([0,1], [0,1], "k--", label="Perfect Calibration")
    plt.plot(confs, accs, "o-", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

def plot_uncertainty_error(uncertainty, errors, out_path):
    plt.figure(figsize=(6, 6))
    # Boxplot of uncertainty for Correct vs Wrong
    u_correct = uncertainty[errors == 0]
    u_wrong = uncertainty[errors == 1]
    
    plt.boxplot([u_correct, u_wrong], labels=['Correct', 'Wrong'])
    plt.ylabel("Epistemic Uncertainty")
    plt.title("Uncertainty vs Prediction Error")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

def plot_coverage_accuracy(probs, labels, uncertainty, out_path):
    """
    Plot Accuracy vs Coverage.
    We rank samples by uncertainty (low to high) or confidence (high to low).
    Here we use confidence (max prob).
    """
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels)
    
    # Sort by confidence descending
    sorted_indices = np.argsort(-confidences)
    sorted_correct = correct[sorted_indices]
    
    accuracies = []
    coverages = np.linspace(0.01, 1.0, 100)
    
    n = len(labels)
    for cov in coverages:
        n_samples = int(cov * n)
        if n_samples == 0:
            accuracies.append(1.0) # convention
        else:
            acc = sorted_correct[:n_samples].mean()
            accuracies.append(acc)
            
    plt.figure(figsize=(6, 6))
    plt.plot(coverages, accuracies, label="Sorted by Confidence")
    plt.xlabel("Coverage (Fraction of data retained)")
    plt.ylabel("Accuracy on retained data")
    plt.title("Accuracy vs Coverage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(probs, labels, out_path, class_names=None):
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, preds)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(cm_norm, cmap='Blues', xticklabels=5, yticklabels=5) # skip some labels if many classes
    else:
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_robustness_curves(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"Robustness CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Unique test types
    test_types = df['test_type'].unique()
    
    for tt in test_types:
        if tt == 'domain_shift': continue 
        
        subset = df[df['test_type'] == tt]
        
        # Sort by param
        subset = subset.sort_values(by='param')
        
        # Plot Acc and Uncertainty vs Param
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Corruption Severity (Param)')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(subset['param'], subset['acc'], color=color, marker='o', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx() 
        color = 'tab:red'
        ax2.set_ylabel('Uncertainty', color=color)
        ax2.plot(subset['param'], subset['uncertainty'], color=color, marker='x', linestyle='--', label='Uncertainty')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Robustness: {tt}")
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, f"robustness_{tt}.png"))
        plt.close()
    
    print(f"Robustness plots saved to {out_dir}")

def make_plots():
    parser = argparse.ArgumentParser(description='Generate calibration and analysis plots')
    parser.add_argument('--mc_file', type=str, help='Path to MC dropout .npz output', default=None)
    parser.add_argument('--robustness_file', type=str, help='Path to robustness .csv output', default=None)
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Calibration / Coverage / Confusion Analysis
    if args.mc_file and os.path.exists(args.mc_file):
        print(f"Loading {args.mc_file}...")
        data = np.load(args.mc_file)
        probs = data['p_mean']
        uncertainty = data['u_epistemic']
        labels = data['y_true']
        
        # ECE
        ece, (accs, confs, counts) = compute_ece(probs, labels)
        print(f"MC Dropout ECE: {ece:.4f}")
        
        with open(os.path.join(args.out_dir, "ece.json"), "w") as f:
            json.dump({"ece": ece}, f)
            
        # Standard calibration plots
        plot_reliability_diagram(accs, confs, ece, os.path.join(args.out_dir, "reliability.png"))
        
        # Uncertainty vs Error
        preds = np.argmax(probs, axis=1)
        errors = (preds != labels).astype(int)
        plot_uncertainty_error(uncertainty, errors, os.path.join(args.out_dir, "uncertainty_error.png"))
        
        # Coverage vs Accuracy
        plot_coverage_accuracy(probs, labels, uncertainty, os.path.join(args.out_dir, "coverage_accuracy.png"))
        
        # Confusion Matrix
        plot_confusion_matrix(probs, labels, os.path.join(args.out_dir, "confusion_matrix.png"))
        
    elif args.mc_file:
         print(f"Warning: --mc_file was provided but not found at {args.mc_file}")

    # 2. Robustness Analysis
    if args.robustness_file:
        plot_robustness_curves(args.robustness_file, args.out_dir)
    
    print(f"All plots saved to {args.out_dir}")

if __name__ == "__main__":
    make_plots()
