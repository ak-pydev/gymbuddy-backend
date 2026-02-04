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
                s_data = s_data[:, :, :self.target_channels]
            
            # Update C
            T, J, C = s_data.shape
        
        # 1. Temporal Sampling / Padding
        s_data = self._resample(s_data, self.target_frames)
        
        # 2. Normalization (Adaptive)
        s_data = self._normalize(s_data)
        
        # 3. Joint Mapping (COCO -> NTU)
        # NTU model expects 25 joints with specific order.
        # COCO has 17. We must map them.
        T, J, C = s_data.shape
        if J == 17:
             s_data = self._map_coco_to_ntu(s_data)
             J = 25 # Now it is 25
        
        # 4. Joint Padding (fallback if not 17 but still < 25, though likely already handled)
        if J < 25:
             diff = 25 - J
             padding = np.zeros((T, diff, C), dtype=s_data.dtype)
             s_data = np.concatenate([s_data, padding], axis=1)

        # Convert to Tensor
        x = torch.from_numpy(s_data).float()
        
        # Label
        y = int(sample.get('label', -1))
        
        # Mask (all ones since we resample)
        mask = torch.ones((self.target_frames, 25), dtype=torch.bool)
        
        return {
            "x": x,
            "y": y,
            "mask": mask
        }

    def _map_coco_to_ntu(self, data):
        """
        Maps 17-joint COCO data to 25-joint NTU format.
        data: (T, 17, C)
        Returns: (T, 25, C)
        """
        T, J, C = data.shape
        ntu_data = np.zeros((T, 25, C), dtype=data.dtype)
        
        # Mapping COCO (Source) -> NTU (Dest)
        # Based on standard topology approximations
        
        # COCO Format:
        # 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar
        # 5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow, 9: LWrist, 10: RWrist
        # 11: LHip, 12: RHip, 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle
        
        # NTU (Kinect V2) Format:
        # 0: SpineBase (Pelvis) -> Midpoint of Hips (11, 12)
        # 1: SpineMid -> Midpoint of (Mid-Hips, Mid-Shoulders)
        # 20: SpineShoulder -> Midpoint of Shoulders (5, 6)
        # 2: Neck -> Midpoint(Shoulders) + slight offset? Or just same as SpineShoulder for now
        # 3: Head -> Nose (0)
        
        # Arms
        # 4: LShoulder -> 5
        # 5: LElbow -> 7
        # 6: LWrist -> 9
        # 7: LHand -> (Copy Wrist)
        # 8: RShoulder -> 6
        # 9: RElbow -> 8
        # 10: RWrist -> 10
        # 11: RHand -> (Copy Wrist)
        
        # Legs
        # 12: LHip -> 11
        # 13: LKnee -> 13
        # 14: LAnkle -> 15
        # 15: LFoot -> (Copy Ankle)
        # 16: RHip -> 12
        # 17: RKnee -> 14
        # 18: RAnkle -> 16
        # 19: RFoot -> (Copy Ankle)
        
        # Calculated joints
        l_hip = data[:, 11, :]
        r_hip = data[:, 12, :]
        mid_hip = (l_hip + r_hip) / 2.0
        
        l_sh = data[:, 5, :]
        r_sh = data[:, 6, :]
        mid_sh = (l_sh + r_sh) / 2.0
        
        # 0: SpineBase
        ntu_data[:, 0, :] = mid_hip
        
        # 1: SpineMid (approx)
        ntu_data[:, 1, :] = (mid_hip + mid_sh) / 2.0
        
        # 20: SpineShoulder
        ntu_data[:, 20, :] = mid_sh
        
        # 2: Neck (approx same as SpineShoulder for lack of better info)
        ntu_data[:, 2, :] = mid_sh
        
        # 3: Head (Nose)
        ntu_data[:, 3, :] = data[:, 0, :]
        
        # Left Arm
        ntu_data[:, 4, :] = data[:, 5, :] # LShoulder
        ntu_data[:, 5, :] = data[:, 7, :] # LElbow
        ntu_data[:, 6, :] = data[:, 9, :] # LWrist
        ntu_data[:, 7, :] = data[:, 9, :] # LHand (Copy)
        
        # Right Arm
        ntu_data[:, 8, :] = data[:, 6, :] # RShoulder
        ntu_data[:, 9, :] = data[:, 8, :] # RElbow
        ntu_data[:, 10, :] = data[:, 10, :] # RWrist
        ntu_data[:, 11, :] = data[:, 10, :] # RHand (Copy)
        
        # Left Leg
        ntu_data[:, 12, :] = data[:, 11, :] # LHip
        ntu_data[:, 13, :] = data[:, 13, :] # LKnee
        ntu_data[:, 14, :] = data[:, 15, :] # LAnkle
        ntu_data[:, 15, :] = data[:, 15, :] # LFoot (Copy)
        
        # Right Leg
        ntu_data[:, 16, :] = data[:, 12, :] # RHip
        ntu_data[:, 17, :] = data[:, 14, :] # RKnee
        ntu_data[:, 18, :] = data[:, 16, :] # RAnkle
        ntu_data[:, 19, :] = data[:, 16, :] # RFoot (Copy)
        
        # 21, 22: HandTips (Copy Wrists)
        ntu_data[:, 21, :] = data[:, 9, :]
        ntu_data[:, 23, :] = data[:, 10, :]
        
        # 22, 24: Thumbs (Copy Wrists)
        ntu_data[:, 22, :] = data[:, 9, :]
        ntu_data[:, 24, :] = data[:, 10, :]
        
        return ntu_data

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
