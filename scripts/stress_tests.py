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

def run_stress_test():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint_path = "outputs/ntu120_baseline/checkpoint.pt"
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        return

    # Load Model
    model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=0.5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # Dataset - Use small subset for speed
    full_ds = NTU120Dataset(split='xsub_val', target_frames=60)
    indices = np.random.choice(len(full_ds), 500, replace=False)
    ds = Subset(full_ds, indices)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # 1. Joint Dropout Test
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    d_accs = []
    d_uncs = []
    
    print("Running Joint Dropout Stress Test...")
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
    
    print("Running Jitter Stress Test...")
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
    os.makedirs("outputs/figs", exist_ok=True)
    
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
    plt.savefig("outputs/figs/stress_dropout.png")
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
    
    plt.title("Stress Test: Gaussian Jitter")
    plt.savefig("outputs/figs/stress_jitter.png")
    plt.close()

if __name__ == "__main__":
    run_stress_test()
