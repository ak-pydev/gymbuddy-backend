#!/usr/bin/env python3
"""
Evaluate NTU-pretrained model on gym_2d data using MC Dropout.
Generates uncertainty outputs for domain comparison.

Includes proper skeleton normalization and frame interpolation to match NTU preprocessing.
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from scipy import interpolate

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from gymbuddy.models.transformer import SkeletonTransformer
from gymbuddy.uncertainty.mc_dropout import predict_mc


def interpolate_frames(skeleton, target_frames):
    """
    Interpolate skeleton sequence to target number of frames using linear interpolation.
    
    Args:
        skeleton: (T, J, C) array - original skeleton sequence
        target_frames: int - target number of frames
    
    Returns:
        (target_frames, J, C) array - interpolated skeleton
    """
    original_frames, num_joints, num_coords = skeleton.shape
    
    if original_frames == target_frames:
        return skeleton
    
    # Create interpolation function for each joint coordinate
    original_indices = np.linspace(0, 1, original_frames)
    target_indices = np.linspace(0, 1, target_frames)
    
    interpolated = np.zeros((target_frames, num_joints, num_coords))
    
    for j in range(num_joints):
        for c in range(num_coords):
            f = interpolate.interp1d(original_indices, skeleton[:, j, c], kind='linear')
            interpolated[:, j, c] = f(target_indices)
    
    return interpolated


def normalize_skeleton(skeleton, center_joint=0):
    """
    Normalize skeleton to be centered at a reference joint (typically hip/spine).
    Also scales to unit bounding box.
    
    Args:
        skeleton: (T, J, C) array - skeleton sequence
        center_joint: int - joint index to center on (0 = typically spine/hip)
    
    Returns:
        (T, J, C) array - normalized skeleton
    """
    T, J, C = skeleton.shape
    
    # Center on reference joint for each frame
    center = skeleton[:, center_joint:center_joint+1, :]  # (T, 1, C)
    centered = skeleton - center
    
    # Scale to unit bounding box
    max_val = np.abs(centered).max()
    if max_val > 1e-6:
        centered = centered / max_val
    
    return centered


def load_gym_data(data_path, target_frames=60, normalize=True, center_joint=0):
    """Load gym skeleton data from pickle file with proper preprocessing."""
    print(f"Loading gym data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, dict):
        if 'skeletons' in data:
            skeletons = data['skeletons']
            labels = data['labels']
        elif 'x' in data:
            skeletons = data['x']
            labels = data['y']
        elif 'annotations' in data:
            # Format: {'split': ..., 'annotations': [{'keypoint': array, 'label': int}, ...]}
            annotations = data['annotations']
            print(f"  Found annotations format with {len(annotations)} samples")
            skeletons = [ann['keypoint'] for ann in annotations]
            labels = [ann['label'] for ann in annotations]
        else:
            raise ValueError(f"Unknown dict keys: {data.keys()}")
    elif isinstance(data, tuple):
        skeletons, labels = data
    elif isinstance(data, list):
        # List of (skeleton, label) tuples or list of dicts
        if isinstance(data[0], dict):
            skeletons = [item['keypoint'] for item in data]
            labels = [item['label'] for item in data]
        else:
            skeletons = [item[0] for item in data]
            labels = [item[1] for item in data]
    else:
        raise ValueError(f"Unknown data format: {type(data)}")
    
    skeletons = np.array(skeletons)
    labels = np.array(labels)
    
    print(f"  Loaded {len(skeletons)} samples")
    print(f"  Original skeleton shape: {skeletons.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique classes: {len(np.unique(labels))}")
    
    # Process each skeleton
    processed_skeletons = []
    for i, skel in enumerate(skeletons):
        # Interpolate to target frames
        if skel.shape[0] != target_frames:
            skel = interpolate_frames(skel, target_frames)
        
        # Normalize skeleton
        if normalize:
            skel = normalize_skeleton(skel, center_joint=center_joint)
        
        processed_skeletons.append(skel)
    
    skeletons = np.array(processed_skeletons)
    
    print(f"  Processed skeleton shape: {skeletons.shape}")
    print(f"  Preprocessing: frames={target_frames}, normalize={normalize}")
    
    return skeletons, labels


def eval_gym():
    parser = argparse.ArgumentParser(description='Evaluate on gym_2d data')
    parser.add_argument('--gym_data', type=str, required=True, help='Path to gym_2d.pkl')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to NTU-pretrained checkpoint')
    parser.add_argument('--out_file', type=str, required=True, help='Path to save MC outputs (.npz)')
    parser.add_argument('--num_classes', type=int, default=120, help='Number of classes in model')
    parser.add_argument('--n_passes', type=int, default=20, help='Number of MC passes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--target_frames', type=int, default=60, help='Target frame count')
    parser.add_argument('--no_normalize', action='store_true', help='Skip skeleton normalization')
    parser.add_argument('--center_joint', type=int, default=0, help='Joint index for centering')
    parser.add_argument('--debug', action='store_true', help='Use small subset')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
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
    model = SkeletonTransformer(
        num_classes=args.num_classes, 
        d_model=256, nhead=4, num_layers=4, dropout=0.5
    )
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    
    # Load gym data with preprocessing
    skeletons, labels = load_gym_data(
        args.gym_data, 
        target_frames=args.target_frames,
        normalize=not args.no_normalize,
        center_joint=args.center_joint
    )
    
    if args.debug:
        print("DEBUG MODE: Using subset of 50 samples")
        indices = np.random.choice(len(skeletons), min(50, len(skeletons)), replace=False)
        skeletons = skeletons[indices]
        labels = labels[indices]
    
    # Create dataloader
    X_tensor = torch.tensor(skeletons, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Run MC dropout
    print(f"Running MC Dropout with {args.n_passes} passes...")
    probs, uncertainty, labels_out = predict_mc(model, loader, n_passes=args.n_passes, device=device)
    
    # Compute predictions and metrics
    y_pred = np.argmax(probs, axis=1)
    accuracy = (y_pred == labels_out).mean()
    
    # Uncertainty analysis
    correct_mask = (y_pred == labels_out)
    u_correct = uncertainty[correct_mask].mean() if correct_mask.sum() > 0 else 0
    u_incorrect = uncertainty[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Samples: {len(labels_out)}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Mean uncertainty: {uncertainty.mean():.6f}")
    print(f"  Std uncertainty: {uncertainty.std():.6f}")
    print(f"  Uncertainty (correct): {u_correct:.6f}")
    print(f"  Uncertainty (incorrect): {u_incorrect:.6f}")
    print(f"  Uncertainty separation: {u_incorrect - u_correct:.6f}")
    print(f"{'='*50}")
    
    # Save
    print(f"Saving to {args.out_file}...")
    np.savez(args.out_file,
             p_mean=probs,
             u_epistemic=uncertainty,
             y_true=labels_out,
             y_pred=y_pred,
             # Metadata
             accuracy=accuracy,
             mean_uncertainty=uncertainty.mean(),
             std_uncertainty=uncertainty.std(),
             n_passes=args.n_passes)
    
    print("Done!")


if __name__ == "__main__":
    eval_gym()

