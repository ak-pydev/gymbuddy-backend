import numpy as np
import matplotlib.pyplot as plt
import os

def run_sweep():
    results_path = "outputs/ntu120_baseline/mc_results.npz"
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}. Run eval_mc_dropout.py first.")
        return
    
    data = np.load(results_path)
    probs = data['probs']
    uncertainty = data['uncertainty']
    labels = data['labels']
    
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels)
    
    # Sweep thresholds
    # We want to cover from 100% data to 0% data
    # Sort uncertainty
    sorted_u = np.sort(uncertainty)
    
    # Pick 50 thresholds
    thresholds = np.unique(np.percentile(sorted_u, np.linspace(0, 100, 50)))
    
    total_samples = len(labels)
    
    coverages = []
    risks = [] # Error on covered
    accuracies = [] # Accuracy on covered
    
    for t in thresholds:
        # Policy: keep if u <= t
        idx = uncertainty <= t
        n_covered = np.sum(idx)
        
        if n_covered == 0:
            continue
            
        coverage = n_covered / total_samples
        acc = np.mean(correct[idx])
        risk = 1.0 - acc
        
        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(acc)
        
    # Plot Coverage vs Risk
    os.makedirs("outputs/figs", exist_ok=True)
    
    plt.figure()
    plt.plot(coverages, risks, "b-", linewidth=2)
    plt.xlabel("Coverage (Fraction of data retained)")
    plt.ylabel("Risk (Error rate on retained)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, max(risks) + 0.1)
    plt.savefig("outputs/figs/coverage_risk.png")
    plt.close()
    
    print("Gating Sweep Complete.")
    print(f"Coverage 1.0 -> Risk: {risks[-1]:.4f}")
    # Find risk at 0.5 coverage
    # Interpolate? Or nearest.
    idx_50 = np.abs(np.array(coverages) - 0.5).argmin()
    print(f"Coverage ~{coverages[idx_50]:.2f} -> Risk: {risks[idx_50]:.4f}")

if __name__ == "__main__":
    run_sweep()
