def run_sweep():
    results_path = "outputs/uncertainty/ntu120_xsub_mc.npz"
    output_dir = "outputs/gating"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}. Run eval_mc_dropout.py first.")
        return
    
    data = np.load(results_path)
    # Keys: p_mean, u_epistemic, y_true...
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
        # Policy: keep if u <= t
        keep_mask = uncertainty <= t
        n_covered = np.sum(keep_mask)
        
        if n_covered == 0:
            continue
            
        coverage = n_covered / total_samples
        
        # Risk: Error rate on covered
        risk = np.mean(incorrect[keep_mask])
        
        # Wrong-and-spoke: Fraction of total samples where model was WRONG but CONFIDENT (passed gate)
        # This is a proxy for unsafe actions.
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
    import csv
    csv_path = os.path.join(output_dir, "gating_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "coverage", "risk_on_covered", "unsafe_rate", "accuracy_covered"])
        writer.writeheader()
        writer.writerows(results)
        
    # Plot Coverage vs Risk
    plt.figure()
    plt.plot(coverages, risks, "b-", linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (Error Rate)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, max(risks) + 0.1)
    plt.savefig(os.path.join(output_dir, "coverage_risk_curve.png"))
    plt.close()
    
    print(f"Gating Sweep Complete. Results saved to {output_dir}")

if __name__ == "__main__":
    run_sweep()
