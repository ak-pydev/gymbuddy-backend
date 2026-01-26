import torch
import torch.nn as nn
import numpy as np

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
def predict_mc(model, data_loader, n_passes=20, device='cpu'):
    """
    Perform Monte-Carlo Dropout Prediction.
    
    Args:
        model: PyTorch model.
        data_loader: DataLoader.
        n_passes: Number of forward passes.
        device: 'cpu' or 'cuda'/'mps'.
        
    Returns:
        mean_probs: (N, C) - Mean softmax probabilities.
        uncertainty: (N,) - Predictive entropy or defined metric (here variance or entropy).
        targets: (N,) - Ground truth labels.
    """
    model.eval()
    enable_dropout(model) # Force dropout on
    
    all_targets = []
    all_preds_passes = [] # List of (N, C) arrays
    
    # Collect data first or run passes per batch?
    # Running passes per batch is memory efficient.
    
    mean_probs_list = []
    uncertainty_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            targets_list.append(y.cpu().numpy())
            
            # Run n_passes for this batch
            batch_probs = []
            for _ in range(n_passes):
                out = model(x) # (B, C)
                probs = torch.softmax(out, dim=1)
                batch_probs.append(probs.unsqueeze(0)) # (1, B, C)
                
            # Stack: (n_passes, B, C)
            batch_probs = torch.cat(batch_probs, dim=0)
            
            # Mean: (B, C)
            mean_probs = batch_probs.mean(dim=0)
            
            # Uncertainty: Entropy of mean dist? Or Variance?
            # User asked for "u_epistemic (variance across passes)".
            # Variance of the probabilities?
            # Usually: Var[p] = E[p^2] - (E[p])^2
            # We can sum variance over classes or take max var.
            # "scalar per sample".
            # Common metric: Predictive Entropy H(p_mean).
            # But if user insists on "variance":
            # Let's compute average variance across classes for each sample.
            variance = batch_probs.var(dim=0).mean(dim=1) # (B,)
            
            mean_probs_list.append(mean_probs.cpu().numpy())
            uncertainty_list.append(variance.cpu().numpy())
            
    return np.concatenate(mean_probs_list), np.concatenate(uncertainty_list), np.concatenate(targets_list)
