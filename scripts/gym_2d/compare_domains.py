#!/usr/bin/env python3
"""
Compare NTU and gym_2d domain results.
Generates side-by-side plots for paper including:
- Uncertainty distributions
- Reliability diagrams (ECE)
- Accuracy-coverage curves
- Risk-coverage curves
- Combined safety metrics table
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import csv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error with bin details."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    ece = 0.0
    total_samples = len(labels)
    accs, confs, counts = [], [], []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            acc_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * np.sum(in_bin) / total_samples
            accs.append(acc_in_bin)
            confs.append(conf_in_bin)
            counts.append(np.sum(in_bin))
    
    return ece, accs, confs, counts


def compute_coverage_metrics(probs, labels, uncertainty):
    """
    Compute accuracy and risk at different coverage levels.
    Returns coverages, accuracies, and risks arrays.
    """
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels)
    incorrect = ~correct
    
    # Sort by uncertainty (low to high = most confident first)
    sorted_indices = np.argsort(uncertainty)
    sorted_correct = correct[sorted_indices]
    sorted_incorrect = incorrect[sorted_indices]
    
    coverages = np.linspace(0.01, 1.0, 100)
    accuracies = []
    risks = []
    wrong_spoke_rates = []
    
    n = len(labels)
    
    for cov in coverages:
        n_samples = int(cov * n)
        if n_samples == 0:
            accuracies.append(1.0)
            risks.append(0.0)
            wrong_spoke_rates.append(0.0)
        else:
            acc = sorted_correct[:n_samples].mean()
            risk = sorted_incorrect[:n_samples].mean()
            wrong_spoke = sorted_incorrect[:n_samples].sum() / n  # As fraction of total
            
            accuracies.append(acc)
            risks.append(risk)
            wrong_spoke_rates.append(wrong_spoke)
    
    return coverages, np.array(accuracies), np.array(risks), np.array(wrong_spoke_rates)


def compute_safety_table(probs, labels, uncertainty, name, target_coverages=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]):
    """Compute safety metrics at fixed coverage points."""
    coverages, accuracies, risks, wrong_spoke_rates = compute_coverage_metrics(probs, labels, uncertainty)
    
    table = []
    for target_cov in target_coverages:
        idx = np.argmin(np.abs(coverages - target_cov))
        table.append({
            'dataset': name,
            'target_coverage': target_cov,
            'actual_coverage': coverages[idx],
            'accuracy': accuracies[idx],
            'risk': risks[idx],
            'wrong_spoke_rate': wrong_spoke_rates[idx]
        })
    
    return table


def compare_domains():
    parser = argparse.ArgumentParser(description='Compare NTU vs gym domains')
    parser.add_argument('--ntu_mc', type=str, required=True, help='Path to NTU MC outputs')
    parser.add_argument('--gym_mc', type=str, required=True, help='Path to gym MC outputs')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--ntu_label', type=str, default='NTU-120 (ID)', help='NTU label for plots')
    parser.add_argument('--gym_label', type=str, default='Gym (Near-OOD)', help='Gym label for plots')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    print("Loading NTU results...")
    ntu_data = np.load(args.ntu_mc)
    ntu_probs = ntu_data['p_mean']
    ntu_uncertainty = ntu_data['u_epistemic']
    ntu_labels = ntu_data['y_true']
    
    print("Loading gym results...")
    gym_data = np.load(args.gym_mc)
    gym_probs = gym_data['p_mean']
    gym_uncertainty = gym_data['u_epistemic']
    gym_labels = gym_data['y_true']
    
    # Compute metrics
    ntu_ece, ntu_accs, ntu_confs, _ = compute_ece(ntu_probs, ntu_labels)
    gym_ece, gym_accs, gym_confs, _ = compute_ece(gym_probs, gym_labels)
    
    ntu_preds = np.argmax(ntu_probs, axis=1)
    gym_preds = np.argmax(gym_probs, axis=1)
    ntu_accuracy = (ntu_preds == ntu_labels).mean()
    gym_accuracy = (gym_preds == gym_labels).mean()
    
    # Uncertainty statistics
    ntu_u_correct = ntu_uncertainty[ntu_preds == ntu_labels].mean()
    ntu_u_incorrect = ntu_uncertainty[ntu_preds != ntu_labels].mean()
    gym_u_correct = gym_uncertainty[gym_preds == gym_labels].mean()
    gym_u_incorrect = gym_uncertainty[gym_preds != gym_labels].mean()
    
    print(f"\n{'='*70}")
    print("DOMAIN COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {args.ntu_label:<20} {args.gym_label:<20}")
    print(f"{'-'*70}")
    print(f"{'Samples':<30} {len(ntu_labels):<20} {len(gym_labels):<20}")
    print(f"{'Accuracy':<30} {ntu_accuracy:<20.4f} {gym_accuracy:<20.4f}")
    print(f"{'ECE':<30} {ntu_ece:<20.4f} {gym_ece:<20.4f}")
    print(f"{'Mean Uncertainty':<30} {ntu_uncertainty.mean():<20.6f} {gym_uncertainty.mean():<20.6f}")
    print(f"{'Uncertainty (correct)':<30} {ntu_u_correct:<20.6f} {gym_u_correct:<20.6f}")
    print(f"{'Uncertainty (incorrect)':<30} {ntu_u_incorrect:<20.6f} {gym_u_incorrect:<20.6f}")
    print(f"{'Uncertainty Separation':<30} {ntu_u_incorrect - ntu_u_correct:<20.6f} {gym_u_incorrect - gym_u_correct:<20.6f}")
    print(f"{'='*70}")
    
    # Save summary JSON
    summary = {
        'ntu': {
            'accuracy': float(ntu_accuracy),
            'ece': float(ntu_ece),
            'mean_uncertainty': float(ntu_uncertainty.mean()),
            'std_uncertainty': float(ntu_uncertainty.std()),
            'uncertainty_correct': float(ntu_u_correct),
            'uncertainty_incorrect': float(ntu_u_incorrect),
            'uncertainty_separation': float(ntu_u_incorrect - ntu_u_correct),
            'n_samples': int(len(ntu_labels))
        },
        'gym': {
            'accuracy': float(gym_accuracy),
            'ece': float(gym_ece),
            'mean_uncertainty': float(gym_uncertainty.mean()),
            'std_uncertainty': float(gym_uncertainty.std()),
            'uncertainty_correct': float(gym_u_correct),
            'uncertainty_incorrect': float(gym_u_incorrect),
            'uncertainty_separation': float(gym_u_incorrect - gym_u_correct),
            'n_samples': int(len(gym_labels))
        }
    }
    
    with open(os.path.join(args.out_dir, 'domain_comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ========================
    # Safety Table (Combined)
    # ========================
    ntu_safety = compute_safety_table(ntu_probs, ntu_labels, ntu_uncertainty, 'NTU-120')
    gym_safety = compute_safety_table(gym_probs, gym_labels, gym_uncertainty, 'Gym')
    
    all_safety = ntu_safety + gym_safety
    
    # Save as CSV
    csv_path = os.path.join(args.out_dir, 'safety_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'target_coverage', 'actual_coverage', 
                                                'accuracy', 'risk', 'wrong_spoke_rate'])
        writer.writeheader()
        writer.writerows(all_safety)
    
    # Print safety table
    print(f"\n{'='*90}")
    print("SAFETY METRICS TABLE (Compare gating effectiveness)")
    print(f"{'='*90}")
    print(f"{'Dataset':<12} {'Coverage':<12} {'Accuracy':<12} {'Risk':<12} {'Wrong-Spoke':<15}")
    print(f"{'-'*90}")
    for row in all_safety:
        print(f"{row['dataset']:<12} {row['actual_coverage']:>10.1%} {row['accuracy']:>10.1%} {row['risk']:>10.1%} {row['wrong_spoke_rate']:>13.2%}")
    print(f"{'='*90}")
    
    # ========================
    # Plot 1: Uncertainty Distributions
    # ========================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(ntu_uncertainty, bins=50, alpha=0.6, label=args.ntu_label, density=True, color='tab:blue')
    ax.hist(gym_uncertainty, bins=50, alpha=0.6, label=args.gym_label, density=True, color='tab:orange')
    
    ax.axvline(ntu_uncertainty.mean(), color='tab:blue', linestyle='--', linewidth=2, label=f'NTU mean')
    ax.axvline(gym_uncertainty.mean(), color='tab:orange', linestyle='--', linewidth=2, label=f'Gym mean')
    
    ax.set_xlabel('Epistemic Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Uncertainty Distribution: ID vs Near-OOD', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'uncertainty_distributions.png'), dpi=150)
    plt.close()
    
    # ========================
    # Plot 2: Reliability Diagrams Side-by-Side
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # NTU
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[0].bar(ntu_confs, ntu_accs, width=0.08, alpha=0.7, color='tab:blue', edgecolor='black')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'{args.ntu_label}\nECE = {ntu_ece:.4f}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    
    # Gym
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[1].bar(gym_confs, gym_accs, width=0.08, alpha=0.7, color='tab:orange', edgecolor='black')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{args.gym_label}\nECE = {gym_ece:.4f}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'reliability_comparison.png'), dpi=150)
    plt.close()
    
    # ========================
    # Plot 3: Accuracy-Coverage Curves
    # ========================
    ntu_cov, ntu_accs_curve, _, _ = compute_coverage_metrics(ntu_probs, ntu_labels, ntu_uncertainty)
    gym_cov, gym_accs_curve, _, _ = compute_coverage_metrics(gym_probs, gym_labels, gym_uncertainty)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(ntu_cov, ntu_accs_curve, '-', color='tab:blue', linewidth=2, label=args.ntu_label)
    ax.plot(gym_cov, gym_accs_curve, '-', color='tab:orange', linewidth=2, label=args.gym_label)
    
    ax.set_xlabel('Coverage (Fraction Retained)', fontsize=12)
    ax.set_ylabel('Accuracy on Retained', fontsize=12)
    ax.set_title('Accuracy-Coverage Curves\n(Steeper = Harder Domain)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'coverage_accuracy_comparison.png'), dpi=150)
    plt.close()
    
    # ========================
    # Plot 4: Risk-Coverage Curves (NEW - Critical for safety story)
    # ========================
    _, _, ntu_risks, ntu_wrong_spoke = compute_coverage_metrics(ntu_probs, ntu_labels, ntu_uncertainty)
    _, _, gym_risks, gym_wrong_spoke = compute_coverage_metrics(gym_probs, gym_labels, gym_uncertainty)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(ntu_cov, ntu_risks, '-', color='tab:blue', linewidth=2, label=f'{args.ntu_label} Risk')
    ax.plot(gym_cov, gym_risks, '-', color='tab:orange', linewidth=2, label=f'{args.gym_label} Risk')
    
    ax.set_xlabel('Coverage (Fraction Retained)', fontsize=12)
    ax.set_ylabel('Risk (Error Rate on Retained)', fontsize=12)
    ax.set_title('Risk-Coverage Curves\n(Lower = Safer with Gating)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'coverage_risk_comparison.png'), dpi=150)
    plt.close()
    
    # ========================
    # Plot 5: Wrong-and-Spoke Rate Comparison (NEW)
    # ========================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(ntu_cov, ntu_wrong_spoke * 100, '-', color='tab:blue', linewidth=2, label=args.ntu_label)
    ax.plot(gym_cov, gym_wrong_spoke * 100, '-', color='tab:orange', linewidth=2, label=args.gym_label)
    
    ax.set_xlabel('Coverage (Fraction Retained)', fontsize=12)
    ax.set_ylabel('Wrong-and-Spoke Rate (%)', fontsize=12)
    ax.set_title('Wrong-and-Spoke Rate vs Coverage\n(Incorrect predictions that passed gating)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'wrong_spoke_comparison.png'), dpi=150)
    plt.close()
    
    # ========================
    # Plot 6: Uncertainty vs Error Box Plot Comparison
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # NTU
    ntu_correct_mask = (ntu_preds == ntu_labels)
    axes[0].boxplot([ntu_uncertainty[ntu_correct_mask], ntu_uncertainty[~ntu_correct_mask]], 
                    labels=['Correct', 'Incorrect'])
    axes[0].set_ylabel('Epistemic Uncertainty', fontsize=12)
    axes[0].set_title(f'{args.ntu_label}\nUncertainty by Prediction Correctness', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Gym
    gym_correct_mask = (gym_preds == gym_labels)
    axes[1].boxplot([gym_uncertainty[gym_correct_mask], gym_uncertainty[~gym_correct_mask]], 
                    labels=['Correct', 'Incorrect'])
    axes[1].set_ylabel('Epistemic Uncertainty', fontsize=12)
    axes[1].set_title(f'{args.gym_label}\nUncertainty by Prediction Correctness', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'uncertainty_error_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\nAll comparison plots saved to {args.out_dir}")
    print("Generated files:")
    print("  - domain_comparison.json")
    print("  - safety_table.csv")
    print("  - uncertainty_distributions.png")
    print("  - reliability_comparison.png")
    print("  - coverage_accuracy_comparison.png")
    print("  - coverage_risk_comparison.png (NEW)")
    print("  - wrong_spoke_comparison.png (NEW)")
    print("  - uncertainty_error_comparison.png")


if __name__ == "__main__":
    compare_domains()

