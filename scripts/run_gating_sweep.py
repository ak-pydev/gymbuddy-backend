import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

def run_sweep():
    parser = argparse.ArgumentParser(description='Run gating sweep and generate risk-coverage curve')
    parser.add_argument('--mc_file', type=str, required=True, help='Path to MC dropout .npz output')
    parser.add_argument('--out_csv', type=str, required=True, help='Path to save sweep CSV')
    parser.add_argument('--out_fig', type=str, required=True, help='Path to save coverage-risk figure')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for additional artifacts')
    parser.add_argument('--dataset_name', type=str, default='dataset', help='Dataset name for labeling')
    args = parser.parse_args()

    results_path = args.mc_file
    csv_path = args.out_csv
    fig_path = args.out_fig
    out_dir = args.out_dir or os.path.dirname(csv_path)
    
    # Create parent dirs if needed
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
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
    thresholds = np.unique(np.percentile(sorted_u, np.linspace(0, 100, 100)))
    
    total_samples = len(labels)
    
    # Store results for CSV
    results = []
    
    coverages = []
    risks = []
    wrong_spoke_rates = []
    
    for t in thresholds:
        keep_mask = uncertainty <= t
        n_covered = np.sum(keep_mask)
        
        if n_covered == 0:
            continue
            
        coverage = n_covered / total_samples
        risk = np.mean(incorrect[keep_mask])
        
        # Wrong-and-spoke: incorrect predictions that passed gating
        n_wrong_spoke = np.sum(incorrect & keep_mask)
        wrong_spoke_rate = n_wrong_spoke / total_samples
        
        acc_covered = np.mean(correct[keep_mask])
        
        coverages.append(coverage)
        risks.append(risk)
        wrong_spoke_rates.append(wrong_spoke_rate)
        
        results.append({
            "threshold": t,
            "coverage": coverage,
            "risk_on_covered": risk,
            "wrong_spoke_rate": wrong_spoke_rate,
            "accuracy_covered": acc_covered,
            "n_covered": int(n_covered),
            "n_wrong_spoke": int(n_wrong_spoke)
        })
        
    # Save CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["threshold", "coverage", "risk_on_covered", "wrong_spoke_rate", 
                      "accuracy_covered", "n_covered", "n_wrong_spoke"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Sweep results saved to {csv_path}")

    # ========================
    # Safety Metrics at Fixed Coverage Points
    # ========================
    fixed_coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    safety_table = []
    
    for target_cov in fixed_coverages:
        # Find closest coverage point
        best_idx = None
        best_diff = float('inf')
        for i, c in enumerate(coverages):
            diff = abs(c - target_cov)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        
        if best_idx is not None:
            r = results[best_idx]
            safety_table.append({
                "target_coverage": target_cov,
                "actual_coverage": r["coverage"],
                "accuracy": r["accuracy_covered"],
                "risk": r["risk_on_covered"],
                "wrong_spoke_rate": r["wrong_spoke_rate"],
                "threshold": r["threshold"]
            })
    
    # Save safety table as JSON
    safety_json_path = os.path.join(out_dir, f"{args.dataset_name}_safety_table.json")
    with open(safety_json_path, "w") as f:
        json.dump(safety_table, f, indent=2)
    print(f"Safety table saved to {safety_json_path}")
    
    # Print safety table for paper
    print("\n" + "="*70)
    print(f"SAFETY METRICS TABLE - {args.dataset_name}")
    print("="*70)
    print(f"{'Coverage':>10} {'Accuracy':>10} {'Risk':>10} {'Wrong-Spoke':>12} {'Threshold':>12}")
    print("-"*70)
    for row in safety_table:
        print(f"{row['actual_coverage']:>10.1%} {row['accuracy']:>10.1%} {row['risk']:>10.1%} {row['wrong_spoke_rate']:>12.2%} {row['threshold']:>12.4f}")
    print("="*70 + "\n")

    # ========================
    # Plot 1: Coverage vs Risk
    # ========================
    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, "b-", linewidth=2, label="Risk (Error Rate)")
    plt.plot(coverages, wrong_spoke_rates, "r--", linewidth=2, label="Wrong-and-Spoke Rate")
    plt.xlabel("Coverage (Fraction of Data Retained)", fontsize=12)
    plt.ylabel("Rate", fontsize=12)
    plt.title(f"Risk-Coverage Curve - {args.dataset_name}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, max(max(risks), max(wrong_spoke_rates)) + 0.05)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Risk-coverage plot saved to {fig_path}")

    # ========================
    # Plot 2: Threshold Sensitivity
    # ========================
    thresholds_arr = [r["threshold"] for r in results]
    accs = [r["accuracy_covered"] for r in results]
    covs = [r["coverage"] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Uncertainty Threshold (τ)', fontsize=12)
    ax1.set_ylabel('Accuracy on Covered', color=color1, fontsize=12)
    ax1.plot(thresholds_arr, accs, color=color1, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Coverage', color=color2, fontsize=12)
    ax2.plot(thresholds_arr, covs, color=color2, linewidth=2, linestyle='--', label='Coverage')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1)
    
    plt.title(f"Threshold Sensitivity - {args.dataset_name}", fontsize=14)
    fig.tight_layout()
    
    threshold_fig_path = os.path.join(out_dir, f"{args.dataset_name}_threshold_sensitivity.png")
    plt.savefig(threshold_fig_path, dpi=150)
    plt.close()
    print(f"Threshold sensitivity plot saved to {threshold_fig_path}")

if __name__ == "__main__":
    run_sweep()

