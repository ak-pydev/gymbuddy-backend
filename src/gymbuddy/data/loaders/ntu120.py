import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class NTU120Dataset(Dataset):
    def __init__(self, data_path='data/raw/skeleton/ntu120/ntu120_3d.pkl', target_frames=60):
        """
        Args:
            data_path (str): Path to the .pkl file.
            target_frames (int): Number of frames to sample/pad to.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        self.target_frames = target_frames
        
        print(f"Loading NTU120 data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.samples = data.get('annotations', [])
        # If 'annotations' key missing, try to use data directly if it's a list
        if not self.samples and isinstance(data, list):
            self.samples = data
            
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract skeleton: (M, T, J, C) -> (T, J, C)
        # Use first person (M=0)
        # Some samples might not have 'keypoint'
        skeleton = sample.get('keypoint', None)
        if skeleton is None:
            # Fallback or error? For now, return zeros if missing to prevent crash
            # But normally we expect data.
            skeleton = np.zeros((1, self.target_frames, 25, 3))
        
        # Take first person: (T, J, C)
        # Shape is (M, T, J, C).
        # We assume M dim exists.
        if len(skeleton.shape) == 4:
            s_data = skeleton[0] # (T, J, C)
        else:
            s_data = skeleton # Assume (T, J, C) already if not 4D
            
        T, J, C = s_data.shape
        
        # 1. Temporal Sampling / Padding
        s_data = self._resample(s_data, self.target_frames)
        
        # 2. Normalization
        s_data = self._normalize(s_data)
        
        # Convert to Tensor
        x = torch.from_numpy(s_data).float()
        
        # Label
        # Expect 'label' to be int
        y = int(sample.get('label', -1))
        
        # Mask
        # Since we resampled to fixed size, mask is all ones.
        # If we had padded with zeros, we might want 0s there.
        # But _resample uses interpolation/sampling, so all frames are valid-ish.
        mask = torch.ones((self.target_frames, J), dtype=torch.bool)
        
        return {
            "x": x,
            "y": y,
            "mask": mask
        }

    def _resample(self, data, target_len):
        """
        Resample data (T, J, C) to target_len along T dimension.
        """
        T, J, C = data.shape
        if T == target_len:
            return data
        
        # Uniform sampling indices
        # We use linspace to pick frames.
        # This handles both downsampling (T > target) and upsampling (T < target) roughly
        # usually upsampling with nearest neighbor in this way repeats frames.
        
        if T == 0:
            return np.zeros((target_len, J, C))
            
        indices = np.linspace(0, T - 1, target_len).astype(int)
        
        # Select frames
        new_data = data[indices] # (target_len, J, C)
        return new_data

    def _normalize(self, data):
        """
        Center at pelvis and scale by torso length.
        data: (T, J, C)
        """
        # Joint 0: SpineBase (Pelvis)
        # Joint 20: SpineShoulder
        
        # 1. Center at pelvis (frame-wise)
        # data[:, 0:1, :] is (T, 1, C)
        pelvis = data[:, 0:1, :]
        data = data - pelvis
        
        # 2. Scale by torso length
        # We use average torso length of the sequence to keep scale consistent across frames?
        # Or per frame?
        # Usually average across frames is more robust to noise, but per-frame handles distance changes if not perfect.
        # Let's use average torso length of the first frame or average of all frames.
        # Robust: Average over all frames.
        
        spine_base = data[:, 0, :]   # (T, 3) (should be 0 after centering)
        spine_shoulder = data[:, 20, :] # (T, 3)
        
        dist = np.linalg.norm(spine_shoulder - spine_base, axis=1) # (T,)
        mean_dist = np.mean(dist)
        
        if mean_dist > 1e-6:
            data = data / mean_dist
        
        return data
