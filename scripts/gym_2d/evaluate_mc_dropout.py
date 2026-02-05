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
from torch.utils.data import DataLoader, Dataset
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


def adapt_skeleton_to_ntu(skeleton):
    """
    Adapt gym 2D skeleton (17 joints × 2 coords) to NTU format (25 joints × 3 coords).
    
    Gym 2D uses COCO 17-keypoint format (from pose estimation):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    NTU uses 25 joints with 3D coords. We map available joints and set z=0.
    """
    T = skeleton.shape[0]
    J_in = skeleton.shape[1]
    C_in = skeleton.shape[2]
    
    # If already 25 joints and 3D, return as is
    if J_in == 25 and C_in == 3:
        return skeleton
    
    # Create output skeleton: 25 joints × 3 coords
    ntu_skeleton = np.zeros((T, 25, 3), dtype=skeleton.dtype)
    
    # Map COCO 17 joints to NTU 25 joints (best effort mapping)
    # NTU joint indices and approximate COCO equivalents:
    coco_to_ntu = {
        0: 3,    # Nose -> Head
        5: 4,    # L Shoulder -> L Shoulder  
        6: 8,    # R Shoulder -> R Shoulder
        7: 5,    # L Elbow -> L Elbow
        8: 9,    # R Elbow -> R Elbow
        9: 6,    # L Wrist -> L Wrist
        10: 10,  # R Wrist -> R Wrist
        11: 12,  # L Hip -> L Hip
        12: 16,  # R Hip -> R Hip
        13: 13,  # L Knee -> L Knee
        14: 17,  # R Knee -> R Knee
        15: 14,  # L Ankle -> L Ankle
        16: 18,  # R Ankle -> R Ankle
    }
    
    # Copy mapped joints (x, y), set z=0
    for coco_idx, ntu_idx in coco_to_ntu.items():
        if coco_idx < J_in:
            ntu_skeleton[:, ntu_idx, :C_in] = skeleton[:, coco_idx, :C_in]
            # z = 0 is already set from zeros initialization
    
    # Create spine center (NTU joint 0) from mid-hip
    if J_in >= 13:  # Has both hips
        ntu_skeleton[:, 0, :C_in] = (skeleton[:, 11, :C_in] + skeleton[:, 12, :C_in]) / 2
    
    # Create spine (NTU joint 1) from mid-shoulder
    if J_in >= 7:  # Has both shoulders
        ntu_skeleton[:, 1, :C_in] = (skeleton[:, 5, :C_in] + skeleton[:, 6, :C_in]) / 2
    
    # Create neck (NTU joint 2) between spine and head
    if J_in >= 7:
        mid_shoulder = (skeleton[:, 5, :C_in] + skeleton[:, 6, :C_in]) / 2
        ntu_skeleton[:, 2, :C_in] = mid_shoulder
    
    return ntu_skeleton


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
    
    # Keep as list for now (variable length sequences)
    labels = np.array(labels)
    
    print(f"  Loaded {len(skeletons)} samples")
    print(f"  First skeleton shape: {np.array(skeletons[0]).shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique classes: {len(np.unique(labels))}")
    
    # Process each skeleton individually (handles variable length)
    processed_skeletons = []
    for i, skel in enumerate(skeletons):
        skel = np.array(skel)
        
        # Handle extra dimension if present (e.g., shape (1, T, J, C) -> (T, J, C))
        while skel.ndim > 3:
            skel = skel.squeeze(0)
        
        # If shape is (T, J, C), proceed; if (M, T, J, C) for M persons, take first
        if skel.ndim == 4:
            skel = skel[0]  # Take first person
        
        # Interpolate to target frames
        if skel.shape[0] != target_frames:
            skel = interpolate_frames(skel, target_frames)
        
        # Adapt gym skeleton to NTU format (17 joints 2D -> 25 joints 3D)
        skel = adapt_skeleton_to_ntu(skel)
        
        # Normalize skeleton
        if normalize:
            skel = normalize_skeleton(skel, center_joint=center_joint)
        
        processed_skeletons.append(skel)
    
    skeletons = np.array(processed_skeletons)
    
    print(f"  Processed skeleton shape: {skeletons.shape}")
    print(f"  Preprocessing: frames={target_frames}, normalize={normalize}")
    
    return skeletons, labels


class DictDataset(Dataset):
    """Dataset that returns batches as dicts with 'x' and 'y' keys for predict_mc."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y': self.y[idx]}


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
    
    
    # Create dataloader (using DictDataset for predict_mc compatibility)
    dataset = DictDataset(skeletons, labels)
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

