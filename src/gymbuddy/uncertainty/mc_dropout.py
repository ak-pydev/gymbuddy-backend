import torch
import torch.nn as nn
import numpy as np

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            
def predict_mc(model, data_loader, n_passes=20, device='cpu'):
    """
    Perform Monte-Carlo Dropout Prediction.
    
    Implements Bayesian approximation via dropout as described in:
    Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: 
    Representing model uncertainty in deep learning.
    
    Args:
        model: PyTorch model.
        data_loader: DataLoader.
        n_passes: Number of forward passes.
        device: 'cpu' or 'cuda'/'mps'.
        
    Returns:
        mean_probs: (N, C) - Mean softmax probabilities.
        uncertainty: (N,) - Mutual Information (Epistemic Uncertainty).
        targets: (N,) - Ground truth labels.
    """
    model.eval()
    enable_dropout(model) # Force dropout on
    
    mean_probs_list = []
    uncertainty_list = []
    targets_list = []
    energy_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            targets_list.append(y.cpu().numpy())
            
            # Run n_passes for this batch
            batch_logits = []
            for _ in range(n_passes):
                out = model(x) # (B, C) logits
                batch_logits.append(out.unsqueeze(0)) # (1, B, C)
                
            # Stack: (n_passes, B, C)
            batch_logits = torch.cat(batch_logits, dim=0)
            
            # 1. Energy Score (OOD Detection)
            # Energy(x) = -T * LogSumExp(f(x)/T). We use T=1.
            # We compute Energy per pass and average it (Expected Energy).
            # energy per pass: (n_passes, B)
            energy_per_pass = -torch.logsumexp(batch_logits, dim=2)
            mean_energy = energy_per_pass.mean(dim=0) # (B,)
            
            # 2. Probabilities
            batch_probs = torch.softmax(batch_logits, dim=2)
            mean_probs = batch_probs.mean(dim=0)
            
            # 3. Uncertainty Metrics
            # Predictive Entropy (Total Uncertainty): H(mean_probs)
            epsilon = 1e-10
            predictive_entropy = -(mean_probs * torch.log(mean_probs + epsilon)).sum(dim=1)
            
            # Aleatoric Entropy (Average Entropy of single passes): E[H(probs)]
            entropy_per_pass = -(batch_probs * torch.log(batch_probs + epsilon)).sum(dim=2)
            aleatoric_entropy = entropy_per_pass.mean(dim=0)
            
            # Mutual Information (Epistemic Uncertainty)
            mutual_info = predictive_entropy - aleatoric_entropy
            mutual_info = torch.clamp(mutual_info, min=0.0)
            
            mean_probs_list.append(mean_probs.cpu().numpy())
            uncertainty_list.append(mutual_info.cpu().numpy())
            energy_list.append(mean_energy.cpu().numpy())
            
    return np.concatenate(mean_probs_list), np.concatenate(uncertainty_list), np.concatenate(targets_list), np.concatenate(energy_list)
