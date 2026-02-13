import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from gymbuddy.data.loaders.ntu120 import NTU120Dataset
from gymbuddy.models.transformer import SkeletonTransformer
from gymbuddy.uncertainty.mc_dropout import predict_mc

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
        
        if prop_in_bin > 0:
            acc_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()
            
            ece += np.abs(acc_in_bin - conf_in_bin) * np.sum(in_bin) / total_samples
            
            accs.append(acc_in_bin)
            confs.append(conf_in_bin)
            counts.append(np.sum(in_bin))
            
    return ece, (accs, confs, counts)




def eval_mc(debug=False, out_file=None, data_path=None, checkpoint=None, num_classes=120, split='xsub_val', n_passes=20, dropout=0.5):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Path to checkpoint
    if checkpoint is None:
        checkpoint_path = "/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs/ntu120_xsub_baseline/best.pt" # Default fallback
    else:
        checkpoint_path = checkpoint

    # Output path for uncertainty stats
    if out_file is None:
        output_dir = "outputs/uncertainty"
        os.makedirs(output_dir, exist_ok=True)
        filename = "ntu_mc.npz"
        output_file = os.path.join(output_dir, filename)
    else:
        output_file = out_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}.")
        sys.exit(1)

    # Load Model (Ensure d_model match)
    print(f"Loading model with num_classes={num_classes}, dropout={dropout} from {checkpoint_path}...")
    model = SkeletonTransformer(num_classes=num_classes, d_model=256, nhead=4, num_layers=4, dropout=dropout)
    
    try:
        if device == 'cpu':
             ckpt = torch.load(checkpoint_path, map_location='cpu')
        else:
             ckpt = torch.load(checkpoint_path)
             
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    model.to(device)
    
    # Dataset (Val)
    print(f"Initializing NTU Validation Dataset (split={split})...")
    val_ds = NTU120Dataset(data_path=data_path, split=split, target_frames=60)
    
    if debug:
        print("DEBUG MODE: Using subset of 50 samples")
        indices = np.random.choice(len(val_ds), 50, replace=False)
        val_ds = torch.utils.data.Subset(val_ds, indices)
        
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print(f"Running MC Dropout Prediction (N={n_passes})...")
    probs, uncertainty, labels = predict_mc(model, val_loader, n_passes=n_passes, device=device)
    
    # Compute predictions (y_pred) from mean probabilities
    y_pred = np.argmax(probs, axis=1)
    
    # Save raw results
    print(f"Saving results to {output_file}...")
    try:
        np.savez(output_file, 
                 p_mean=probs, 
                 u_epistemic=uncertainty, 
                 y_true=labels, 
                 y_pred=y_pred)
        
        if os.path.exists(output_file):
             print(f"SUCCESS: File created at {output_file} ({os.path.getsize(output_file)} bytes)")
        else:
             print(f"ERROR: np.savez completed but file not found at {output_file}")
             sys.exit(1)
             
    except Exception as e:
        print(f"ERROR saving file: {e}")
        sys.exit(1)
        
    print("Inference Complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run on a small subset')
    parser.add_argument('--out_file', type=str, default=None, help='Path to save output .npz file')
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset pkl')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=120, help='Number of classes (default 120 for NTU)')
    parser.add_argument('--split', type=str, default='xsub_val', help='Dataset split')
    parser.add_argument('--n_passes', type=int, default=20, help='Number of MC passes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    # Support --out as alias for --out_file by creating a widely permissive parser
    # But argparse usually handles prefix matching. Let's just rely on that or user can fix SLURM. 
    # Actually, let's explicitly add --out if out_file is the only "out"
    args = parser.parse_args()
    
    print(f"DEBUG: args={args}")
    print(f"DEBUG: out_file={args.out_file}")
    
    eval_mc(debug=args.debug, out_file=args.out_file,
            data_path=args.data_path, checkpoint=args.checkpoint, num_classes=args.num_classes,
            split=args.split, n_passes=args.n_passes, dropout=args.dropout)

