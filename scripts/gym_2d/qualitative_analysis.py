import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import pickle

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
# Assuming standard skeleton visualization utils exist or we write a simple one
# from gymbuddy.utils.vis import plot_skeleton

def qualitative_analysis(mc_file, gym_data_path, out_dir):
    print(f"Loading MC outputs from {mc_file}...")
    data = np.load(mc_file)
    probs = data['probs']
    uncertainties = data['uncertainties'] # [N] (entropy or variation)
    labels = data['labels']
    
    mean_probs = probs.mean(axis=1)
    preds = mean_probs.argmax(axis=1)
    confidences = mean_probs.max(axis=1)
    
    # Load raw data to visualize
    print(f"Loading raw data from {gym_data_path}...")
    with open(gym_data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Assuming correspondence: raw_data['xsub_val'] matches the MC output order
    # Verify lengths
    if len(preds) != len(raw_data['xsub_val']):
        print(f"Warning: Length mismatch. Preds: {len(preds)}, Data: {len(raw_data['xsub_val'])}")
        # Proceeding might be dangerous if indices don't align.
        # Assuming they align for now via standard DataLoader order.
    
    val_samples = raw_data['xsub_val']
    
    # 1. "Saved" Cases: Wrong Prediction but High Uncertainty (Gating WOULD reject)
    # condition: pred != label AND uncertainty > threshold (e.g. median or 75th percentile)
    threshold = np.percentile(uncertainties, 75)
    
    wrong_mask = (preds != labels)
    saved_mask = wrong_mask & (uncertainties > threshold)
    saved_indices = np.where(saved_mask)[0]
    
    # 2. "Fail" Cases: Wrong Prediction and Low Uncertainty (Gating FAILS to reject)
    fail_mask = wrong_mask & (uncertainties < np.percentile(uncertainties, 25))
    fail_indices = np.where(fail_mask)[0]
    
    print(f"Found {len(saved_indices)} 'Saved' cases (Wrong but Uncertain).")
    print(f"Found {len(fail_indices)} 'Fail' cases (Wrong and Confident).")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot helper (placeholder for actual skeleton plot)
    def plot_sample(idx, save_name):
        sample = val_samples[idx]
        # sample typically has 'frame', 'label', 'body'
        # Let's assume 'body' is [C, T, V, M] or similar.
        
        # Here we just create a dummy "frame" or text representation 
        # because we might not have a renderer installed.
        # Ideally, we verify with the user if they have a renderer.
        
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Sample {idx}\nTrue: {labels[idx]}\nPred: {preds[idx]}\nConf: {confidences[idx]:.2f}\nUnc: {uncertainties[idx]:.4f}", 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.title(save_name)
        plt.savefig(os.path.join(out_dir, f"{save_name}.png"))
        plt.close()
        
    # Save top 3 of each
    for i in range(min(3, len(saved_indices))):
        idx = saved_indices[i]
        plot_sample(idx, f"saved_case_{i}_class_{labels[idx]}")
        
    for i in range(min(3, len(fail_indices))):
        idx = fail_indices[i]
        plot_sample(idx, f"fail_case_{i}_class_{labels[idx]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mc_file', type=str, required=True)
    parser.add_argument('--gym_data', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    out_dir = os.path.join(args.out_dir, 'training_analysis', 'qualitative')
    os.makedirs(out_dir, exist_ok=True)
    
    qualitative_analysis(args.mc_file, args.gym_data, out_dir)
