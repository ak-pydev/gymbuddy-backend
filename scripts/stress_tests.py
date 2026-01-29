import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
from gymbuddy.models.transformer import SkeletonTransformer
from gymbuddy.uncertainty.mc_dropout import predict_mc

def apply_joint_dropout(x, p=0.0):
    """ x: (B, T, J, C) """
    if p <= 0: return x
    B, T, J, C = x.shape
    # Mask per sample per joint? Or per frame?
    # Usually "joint dropout" means a joint is missing for the whole sequence or frames.
    # Let's do per-frame joint dropout (simulating occlusion) or per-sample.
    # Per-sample is more realistic for "sensor fail".
    mask = torch.rand(B, 1, J, 1, device=x.device) > p
    return x * mask.float()

def apply_jitter(x, std=0.0):
    if std <= 0: return x
    noise = torch.randn_like(x) * std
    return x + noise

def run_stress_test(debug=False, checkpoint_path=None, out_dir=None):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    if checkpoint_path is None:
        checkpoint_path = "/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs/ntu120_xsub_baseline/best.pt"
    if out_dir is None:
        out_dir = "/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # Load Model
    model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=0.5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # Dataset - Use small subset for speed
    full_ds = NTU120Dataset(split='xsub_val', target_frames=60)
    n_samples = 50 if debug else 500
    if len(full_ds) < n_samples: n_samples = len(full_ds)
    
    indices = np.random.choice(len(full_ds), n_samples, replace=False)
    ds = Subset(full_ds, indices)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # 1. Joint Dropout Test
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    d_accs = []
    d_uncs = []
    
    print(f"Running Joint Dropout Stress Test (N={n_samples})...")
    for p in probs:
        # Wrap loader or transform batch manually?
        # Helper to run predict_mc with corruption
        # We need to inject corruption inside prediction loop or wrap dataset.
        # Easiest: Wrap predict_mc or modify it? 
        # Actually `predict_mc` takes loader. I'll transform data in loader?
        # No, I'll modify the loop here.
        
        # We can re-implement simple MC loop here to apply transform
        all_acc = []
        all_u = []
        
        model.eval()
        # Enable dropout for MC
        indices_dropout = [m for m in model.modules() if m.__class__.__name__.startswith('Dropout')]
        for m in indices_dropout: m.train()
            
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                # Corrupt
                x_corr = apply_joint_dropout(x, p=p)
                
                # Predict
                batch_probs = []
                for _ in range(10): # 10 passes
                    out = model(x_corr)
                    batch_probs.append(torch.softmax(out, dim=1).unsqueeze(0))
                
                mean_probs = torch.cat(batch_probs, dim=0).mean(dim=0)
                variance = torch.cat(batch_probs, dim=0).var(dim=0).mean(dim=1)
                
                preds = mean_probs.argmax(dim=1)
                acc = (preds == y).float().mean().item()
                
                all_acc.append(acc)
                all_u.append(variance.mean().item())
                
        avg_acc = np.mean(all_acc)
        avg_u = np.mean(all_u)
        d_accs.append(avg_acc)
        d_uncs.append(avg_u)
        print(f"  p={p}: Acc={avg_acc:.4f}, Unc={avg_u:.4f}")

    # 2. Jitter Test
    stds = [0.0, 0.01, 0.02, 0.05, 0.1]
    j_accs = []
    j_uncs = []
    
    print(f"Running Jitter Stress Test (N={n_samples})...")
    for s in stds:
        all_acc = []
        all_u = []
        
        # Enable dropout for MC
        indices_dropout = [m for m in model.modules() if m.__class__.__name__.startswith('Dropout')]
        for m in indices_dropout: m.train()

        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                x_corr = apply_jitter(x, std=s)
                
                batch_probs = []
                for _ in range(10):
                    out = model(x_corr)
                    batch_probs.append(torch.softmax(out, dim=1).unsqueeze(0))
                
                mean_probs = torch.cat(batch_probs, dim=0).mean(dim=0)
                variance = torch.cat(batch_probs, dim=0).var(dim=0).mean(dim=1)
                
                preds = mean_probs.argmax(dim=1)
                acc = (preds == y).float().mean().item()
                
                all_acc.append(acc)
                all_u.append(variance.mean().item())
                
        avg_acc = np.mean(all_acc)
        avg_u = np.mean(all_u)
        j_accs.append(avg_acc)
        j_uncs.append(avg_u)
        print(f"  std={s}: Acc={avg_acc:.4f}, Unc={avg_u:.4f}")

    # Plots
    # output_dir is passed as argument
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    # Dropout Plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Joint Dropout Probability')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(probs, d_accs, color='tab:blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Uncertainty', color='tab:red')
    ax2.plot(probs, d_uncs, color='tab:red', marker='x')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title("Stress Test: Joint Dropout")
    plt.savefig(os.path.join(figs_dir, "stress_dropout.png"))
    plt.close()

    # Jitter Plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Gaussian Jitter Std')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(stds, j_accs, color='tab:blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Uncertainty', color='tab:red')
    ax2.plot(stds, j_uncs, color='tab:red', marker='x')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # 3. Domain Shift: Evaluation on xset split (Cross-Setup)
    print("Running Domain Shift Evaluation (xset)...")
    try:
        xset_ds = NTU120Dataset(split='xset_val', target_frames=60)
        
        if debug:
             print("DEBUG: Using xset subset")
             indices = np.random.choice(len(xset_ds), 50, replace=False)
             xset_ds = Subset(xset_ds, indices)
             
        xset_loader = DataLoader(xset_ds, batch_size=32, shuffle=False)
        
        all_acc = []
        all_u = []
        
        # MC Eval Loop for xset
        # Enable dropout
        instances_dropout = [m for m in model.modules() if m.__class__.__name__.startswith('Dropout')]
        for m in instances_dropout: m.train()
            
        with torch.no_grad():
            for batch in xset_loader:
                 x = batch['x'].to(device)
                 y = batch['y'].to(device)
                 
                 batch_probs = []
                 for _ in range(10):
                     out = model(x)
                     batch_probs.append(torch.softmax(out, dim=1).unsqueeze(0))
                     
                 mean_probs = torch.cat(batch_probs, dim=0).mean(dim=0)
                 variance = torch.cat(batch_probs, dim=0).var(dim=0).mean(dim=1)
                 
                 preds = mean_probs.argmax(dim=1)
                 acc = (preds == y).float().mean().item()
                 
                 all_acc.append(acc)
                 all_u.append(variance.mean().item())
        
        xset_acc = np.mean(all_acc)
        xset_u = np.mean(all_u)
        print(f"  xSet (OOD): Acc={xset_acc:.4f}, Unc={xset_u:.4f}")
        
        # Save to table
        csv_path = os.path.join(output_dir, "stress_robustness.csv")
        with open(csv_path, "w") as f:
            f.write("test_type,param,acc,uncertainty\n")
            for p, a, u in zip(probs, d_accs, d_uncs):
                f.write(f"joint_dropout,{p},{a:.4f},{u:.4f}\n")
            for s, a, u in zip(stds, j_accs, j_uncs):
                f.write(f"jitter,{s},{a:.4f},{u:.4f}\n")
            f.write(f"domain_shift,xset,{xset_acc:.4f},{xset_u:.4f}\n")
            
    except Exception as e:
        print(f"Skipping xset run (dataset maybe missing): {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run verification with fewer samples')
    parser.add_argument('--checkpoint', type=str, default="/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs/ntu120_xsub_baseline/best.pt", help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default="/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs", help='Output directory')
    args = parser.parse_args()
    
    run_stress_test(debug=args.debug, checkpoint_path=args.checkpoint, out_dir=args.out_dir)
