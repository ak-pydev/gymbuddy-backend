import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def run_sweep():
    parser = argparse.ArgumentParser(description='Run gating sweep and generate risk-coverage curve')
    parser.add_argument('--mc_file', type=str, required=True, help='Path to MC dropout .npz output')
    parser.add_argument('--out_csv', type=str, required=True, help='Path to save sweep CSV')
    parser.add_argument('--out_fig', type=str, required=True, help='Path to save coverage-risk figure')
    args = parser.parse_args()

    results_path = args.mc_file
    csv_path = args.out_csv
    fig_path = args.out_fig
    
    # Create parent dirs if needed
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return
    
    print(f"Loading {results_path}...")
    data = np.load(results_path)
    probs = data['p_mean']
    uncertainty = data['u_epistemic']
    labels = data['y_true']
    
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels)
    incorrect = (preds != labels)
    
    # Sweep thresholds
    sorted_u = np.sort(uncertainty)
    thresholds = np.unique(np.percentile(sorted_u, np.linspace(0, 100, 50)))
    
    total_samples = len(labels)
    
    # Store results for CSV
    results = []
    
    coverages = []
    risks = [] 
    
    for t in thresholds:
        keep_mask = uncertainty <= t
        n_covered = np.sum(keep_mask)
        
        if n_covered == 0:
            continue
            
        coverage = n_covered / total_samples
        risk = np.mean(incorrect[keep_mask])
        
        n_unsafe = np.sum(incorrect & keep_mask)
        unsafe_rate = n_unsafe / total_samples
        
        acc_covered = np.mean(correct[keep_mask])
        
        coverages.append(coverage)
        risks.append(risk)
        
        results.append({
            "threshold": t,
            "coverage": coverage,
            "risk_on_covered": risk,
            "unsafe_rate": unsafe_rate,
            "accuracy_covered": acc_covered
        })
        
    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "coverage", "risk_on_covered", "unsafe_rate", "accuracy_covered"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Sweep results saved to {csv_path}")

    # Plot Coverage vs Risk
    plt.figure()
    plt.plot(coverages, risks, "b-", linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (Error Rate)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, max(risks) + 0.1)
    plt.savefig(fig_path)
    plt.close()
    
    print(f"Plot saved to {fig_path}")

if __name__ == "__main__":
    run_sweep()
