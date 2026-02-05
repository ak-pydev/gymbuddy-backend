#!/usr/bin/env python3
"""
Ablation study for MC Dropout uncertainty estimation.
Compares different numbers of MC passes and evaluates uncertainty quality.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
from gymbuddy.models.transformer import SkeletonTransformer
from gymbuddy.uncertainty.mc_dropout import predict_mc


def compute_calibration_metrics(probs, labels, uncertainty, n_bins=10):
    """Compute ECE and other calibration metrics."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct = (preds == labels)
    
    ece = 0.0
    total_samples = len(labels)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            acc_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * np.sum(in_bin) / total_samples
    
    # Uncertainty-error correlation
    errors = (~correct).astype(float)
    corr = np.corrcoef(uncertainty, errors)[0, 1]
    
    # Mean uncertainty for correct vs incorrect
    mean_u_correct = uncertainty[correct].mean() if correct.sum() > 0 else 0
    mean_u_incorrect = uncertainty[~correct].mean() if (~correct).sum() > 0 else 0
    
    return {
        'ece': float(ece),
        'accuracy': float(correct.mean()),
        'uncertainty_error_corr': float(corr) if not np.isnan(corr) else 0,
        'mean_u_correct': float(mean_u_correct),
        'mean_u_incorrect': float(mean_u_incorrect),
        'uncertainty_separation': float(mean_u_incorrect - mean_u_correct)
    }


def run_ablation():
    parser = argparse.ArgumentParser(description='MC Dropout Ablation Study')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset pkl')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--mc_passes', type=str, default='5,10,15,20,30', help='Comma-separated MC pass counts')
    parser.add_argument('--num_classes', type=int, default=120, help='Number of classes')
    parser.add_argument('--debug', action='store_true', help='Use small subset for testing')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = SkeletonTransformer(num_classes=args.num_classes, d_model=256, nhead=4, num_layers=4, dropout=0.5)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    
    # Load data
    print("Loading validation dataset...")
    val_ds = NTU120Dataset(data_path=args.data_path, split='xsub_val', target_frames=60)
    
    if args.debug:
        print("DEBUG MODE: Using subset of 100 samples")
        indices = np.random.choice(len(val_ds), min(100, len(val_ds)), replace=False)
        val_ds = torch.utils.data.Subset(val_ds, indices)
    
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Parse MC passes
    mc_pass_counts = [int(x.strip()) for x in args.mc_passes.split(',')]
    print(f"Testing MC passes: {mc_pass_counts}")
    
    # Run ablation
    results = []
    
    for n_passes in mc_pass_counts:
        print(f"\n{'='*50}")
        print(f"Running MC Dropout with {n_passes} passes...")
        print(f"{'='*50}")
        
        probs, uncertainty, labels = predict_mc(model, val_loader, n_passes=n_passes, device=device)
        
        metrics = compute_calibration_metrics(probs, labels, uncertainty)
        metrics['n_passes'] = n_passes
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  U-Error Correlation: {metrics['uncertainty_error_corr']:.4f}")
        print(f"  Uncertainty Separation: {metrics['uncertainty_separation']:.6f}")
        
        results.append(metrics)
    
    # Save results
    results_path = os.path.join(args.out_dir, 'mc_passes_ablation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate plots
    n_passes_list = [r['n_passes'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    eces = [r['ece'] for r in results]
    separations = [r['uncertainty_separation'] for r in results]
    correlations = [r['uncertainty_error_corr'] for r in results]
    
    # Plot 1: Accuracy and ECE vs MC Passes
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of MC Passes', fontsize=12)
    ax1.set_ylabel('Accuracy', color=color1, fontsize=12)
    ax1.plot(n_passes_list, accuracies, 'o-', color=color1, linewidth=2, markersize=8, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('ECE', color=color2, fontsize=12)
    ax2.plot(n_passes_list, eces, 's--', color=color2, linewidth=2, markersize=8, label='ECE')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('MC Dropout Ablation: Accuracy & Calibration vs Passes', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'ablation_accuracy_ece.png'), dpi=150)
    plt.close()
    
    # Plot 2: Uncertainty Quality Metrics
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(n_passes_list, separations, 'o-', color='tab:green', linewidth=2, markersize=8, label='Uncertainty Separation')
    ax.plot(n_passes_list, correlations, 's--', color='tab:purple', linewidth=2, markersize=8, label='U-Error Correlation')
    
    ax.set_xlabel('Number of MC Passes', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('MC Dropout Ablation: Uncertainty Quality', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'ablation_uncertainty_quality.png'), dpi=150)
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("MC PASSES ABLATION SUMMARY")
    print("="*80)
    print(f"{'Passes':>8} {'Accuracy':>10} {'ECE':>10} {'U-Separation':>14} {'U-Error Corr':>14}")
    print("-"*80)
    for r in results:
        print(f"{r['n_passes']:>8} {r['accuracy']:>10.4f} {r['ece']:>10.4f} {r['uncertainty_separation']:>14.6f} {r['uncertainty_error_corr']:>14.4f}")
    print("="*80)
    
    print(f"\nAll ablation outputs saved to {args.out_dir}")


if __name__ == "__main__":
    run_ablation()
