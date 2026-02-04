import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path

class GymDataset(Dataset):
    def __init__(self, data_path=None, target_frames=60, target_channels=3, split='train'):
        """
        Args:
            data_path (str or Path): Path to the .pkl file. If None, uses DATA_ROOT env var.
            target_frames (int): Number of frames to sample/pad to.
            target_channels (int): Target number of channels (e.g. 3 for x,y,z). Pads if data has fewer.
            split (str): Split name (train/val/start).
        """
        if data_path is None:
            if "DATA_ROOT" in os.environ:
                DATA_ROOT = Path(os.environ["DATA_ROOT"])
                data_path = DATA_ROOT / "gym" / "gym_2d.pkl"
            else:
                 # Default Anvil path
                 data_path = Path("/anvil/scratch/x-akhanal3/ai-gym-buddy/data/raw/skeleton/gym/gym_2d.pkl")
        else:
            data_path = Path(data_path)

        if not data_path.exists():
             # Try fallback to just filename if it was passed as relative
             if (Path("datasets") / data_path).exists():
                 data_path = Path("datasets") / data_path

        assert data_path.exists(), f"File not found: {data_path}"
        
        self.target_frames = target_frames
        self.target_channels = target_channels
        
        print(f"Loading Gym data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        # Gym data structure assumed: list of dicts with 'keypoint', 'label', 'frame_dir' (or similar id)
        # Or a dict with 'annotations' key.
        
        if isinstance(data, dict):
            all_annotations = data.get('annotations', [])
        else:
            all_annotations = []
            
        if not all_annotations and isinstance(data, list):
            all_annotations = data
            
        # Handle Splits if available in the pkl
        # Assume standard 80/20 random split if 'split' key not in data
        # Or if the user provided split names match.
        
        if 'split' in data and split in data['split']:
             split_ids = set(data['split'][split])
             print(f"Filtering for split '{split}' with {len(split_ids)} samples...")
             self.samples = [ann for ann in all_annotations if ann.get('frame_dir') in split_ids]
        else:
            # If no split info in file, we might need manual splitting or just load all.
            # Ideally the pkl should have splits. 
            # If not, let's warn and load all (or maybe implement a deterministic random split here?)
            # For now, load all if split not found, but warn.
            if split is not None and split != 'all':
                 print(f"Warning: Split '{split}' not found in dataset keys. Loading all data (or check if 'split' dict exists). Keys found: {data.keys() if isinstance(data, dict) else 'List'}")
            
            self.samples = all_annotations
            
        print(f"Loaded {len(self.samples)} samples for split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract skeleton: Expect (T, J, C) or (M, T, J, C)
        skeleton = sample.get('keypoint', None)
        if skeleton is None:
             # Try other keys? 'data'?
             skeleton = sample.get('data', np.zeros((1, self.target_frames, 25, 3)))

        # Handle dimensions
        if len(skeleton.shape) == 4:
            s_data = skeleton[0] # (T, J, C) take first person
        else:
            s_data = skeleton
            
        # Ensure it is numpy
        if not isinstance(s_data, np.ndarray):
            s_data = np.array(s_data)

        T, J, C = s_data.shape
        
        # 0. Channel Adaptation (e.g. 2D -> 3D)
        if hasattr(self, 'target_channels') and C != self.target_channels:
            if C < self.target_channels:
                # Pad with zeros
                diff = self.target_channels - C
                # (T, J, C) -> (T, J, C+diff)
                padding = np.zeros((T, J, diff), dtype=s_data.dtype)
                s_data = np.concatenate([s_data, padding], axis=-1)
            elif C > self.target_channels:
                # Crop? Or assume first N are correct.
                s_data = s_data[:, :, :self.target_channels]
            
            # Update C
            T, J, C = s_data.shape

        # 0.5. Joint Padding (e.g. 17 -> 25)
        # NTU model expects 25 joints. If we have 17 (COCO), we pad.
        if J < 25:
            diff = 25 - J
            # (T, J, C) -> (T, J+diff, C)
            padding = np.zeros((T, diff, C), dtype=s_data.dtype)
            s_data = np.concatenate([s_data, padding], axis=1)
            T, J, C = s_data.shape
        
        # 1. Temporal Sampling / Padding
        s_data = self._resample(s_data, self.target_frames)
        
        # 2. Normalization
        s_data = self._normalize(s_data)
        
        # Convert to Tensor
        x = torch.from_numpy(s_data).float()
        
        # Label
        y = int(sample.get('label', -1))
        
        # Mask (all ones since we resample)
        mask = torch.ones((self.target_frames, J), dtype=torch.bool)
        
        return {
            "x": x,
            "y": y,
            "mask": mask
        }

    def _resample(self, data, target_len):
        T, J, C = data.shape
        if T == target_len:
            return data
        if T == 0:
            return np.zeros((target_len, J, C))
            
        indices = np.linspace(0, T - 1, target_len).astype(int)
        new_data = data[indices]
        return new_data

    def _normalize(self, data):
        """
        Center at pelvis and scale by torso length.
        data: (T, J, C)
        """
        T, J, C = data.shape
        
        # Determine format based on J
        if J == 17:
            # COCO Format
            # 5: L Shoulder, 6: R Shoulder
            # 11: L Hip, 12: R Hip
            # Pelvis = Midpoint(11, 12)
            # Torso = Midpoint(5, 6)
            
            l_hip = data[:, 11, :]
            r_hip = data[:, 12, :]
            pelvis = (l_hip + r_hip) / 2.0 # (T, C)
            pelvis = pelvis.reshape(T, 1, C)
            
            data = data - pelvis
            
            l_shoulder = data[:, 5, :]
            r_shoulder = data[:, 6, :]
            torso_center = (l_shoulder + r_shoulder) / 2.0
            
            # Spine base (0 after centering) to torso center
            spine_base = np.zeros((T, C)) # It is at 0,0,0
            
            dist = np.linalg.norm(torso_center - spine_base, axis=1)
            mean_dist = np.mean(dist)
            
            if mean_dist > 1e-6:
                data = data / mean_dist
                
        else:
            # Assume NTU / Kinect 25
            # Joint 0: SpineBase (Pelvis)
            # Joint 20: SpineShoulder
            
            pelvis = data[:, 0:1, :]
            data = data - pelvis
            
            spine_base = data[:, 0, :]   
            spine_shoulder = data[:, 20, :] 
            
            dist = np.linalg.norm(spine_shoulder - spine_base, axis=1) 
            mean_dist = np.mean(dist)
            
            if mean_dist > 1e-6:
                data = data / mean_dist
        
        return data
