import torch
import time
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# Mocking model for latency test to avoid dependency hell if possible, or using actual model definition
# Let's try to import the actual model first as it's more accurate.
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
from gymbuddy.models.transformer import SkeletonTransformer

def measure_latency(model, device, n_passes_list, batch_size=1, seq_len=60, num_joints=25, num_channels=2):
    dummy_input = torch.randn(batch_size, seq_len, num_joints, num_channels).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = model(dummy_input)
    
    results = {}
    
    for n_passes in n_passes_list:
        print(f"Measuring latency for {n_passes} MC passes...")
        start_time = time.time()
        n_iters = 50
        
        with torch.no_grad():
            for _ in range(n_iters):
                # Simulate MC Dropout by running forward pass n times
                # In practice, we loop n times or use batch expansion. 
                # Let's assume we loop as that's the standard implementation in mc_dropout.py
                probs_list = []
                for _ in range(n_passes):
                    out = model(dummy_input)
                    probs_list.append(torch.softmax(out, dim=1))
                # Average (simulation of post-processing)
                _ = torch.stack(probs_list).mean(dim=0)
                
        end_time = time.time()
        avg_time = (end_time - start_time) / n_iters
        fps = 1.0 / avg_time
        results[n_passes] = fps
        print(f"  FPS: {fps:.2f}")

    return results

def plot_latency(results, out_dir):
    passes = list(results.keys())
    fps = list(results.values())
    
    plt.figure(figsize=(8, 6))
    plt.plot(passes, fps, 'o-', linewidth=2)
    plt.xlabel('Number of MC Passes')
    plt.ylabel('Inference FPS (Batch Size 1)')
    plt.title('Inference Latency vs MC Dropout Passes')
    plt.grid(True, alpha=0.3)
    
    for i, txt in enumerate(fps):
        plt.annotate(f"{txt:.1f}", (passes[i], fps[i]), xytext=(0, 5), textcoords='offset points', ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'inference_latency.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, required=True)
    args = parser.parse_args()
    
    out_dir = os.path.join(args.outputs_dir, 'training_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    
    # Instantiate model (architecture only, weights don't matter for latency)
    model = SkeletonTransformer(num_classes=120, d_model=256, nhead=4, num_layers=4, dropout=0.1)
    model.to(device)
    model.train() # Enable dropout (though not strictly necessary for simple forward pass timing, but good for correctness if logic changes)
    
    n_passes_list = [1, 5, 10, 20, 50]
    results = measure_latency(model, device, n_passes_list)
    
    plot_latency(results, out_dir)
    print(f"Saved inference_latency.png to {out_dir}")
