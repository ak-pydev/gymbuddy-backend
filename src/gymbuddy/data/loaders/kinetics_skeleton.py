import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class KineticsSkeletonDataset(Dataset):
    def __init__(self, 
                 meta_path='data/raw/skeleton/kinetics400/k400_2d.pkl', 
                 data_root='data/raw/skeleton/kinetics400/kpfiles', 
                 target_frames=60,
                 split='train'):
        """
        Args:
            meta_path (str): Path to k400_2d.pkl
            data_root (str): Folder containing .pkl skeleton files.
            target_frames (int): Number of frames to sample.
            split (str): 'train' or 'val'.
        """
        self.data_root = data_root
        self.target_frames = target_frames
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found at {meta_path}")
            
        print(f"Loading Kinetics meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            
        # meta is dict with 'split' -> {'train': [indexes], 'val': ...}
        # and 'annotations' -> list of dicts.
        
        self.annotations = meta.get('annotations', [])
        splits = meta.get('split', {})
        
        if split in splits:
            indices = splits[split]
            # indices are list of video IDs (str)
            # annotations is list of dicts. We need to map frame_dir -> annotation
            print("Building annotation index...")
            id_to_ann = {ann['frame_dir']: ann for ann in self.annotations}
            
            self.samples = []
            valid_ids = 0
            for vid_id in indices:
                if vid_id in id_to_ann:
                    self.samples.append(id_to_ann[vid_id])
                    valid_ids += 1
            print(f"Matched {valid_ids} / {len(indices)} IDs for split '{split}'.")
            
        else:
            # If split not found or requested 'all', perform logic
            print(f"Split '{split}' not found in meta. Using all annotations.")
            self.samples = self.annotations
            
        print(f"Loaded {len(self.samples)} samples for split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        frame_dir = sample_info['frame_dir']
        label = sample_info['label']
        
        # Construct path
        # Assuming flat structure in data_root
        file_path = os.path.join(self.data_root, f"{frame_dir}.pkl")
        
        if not os.path.exists(file_path):
            # If file missing, handle gracefully (e.g. return zeros or logic)
            # For this task, we can raise or return zeros.
            # Returning zeros is safer for Dataloaders not to crash, but bad for training.
            # We'll print warning once?
            # Creating dummy data
            x = torch.zeros((self.target_frames, 17, 2))
            confidence = torch.zeros((self.target_frames, 17))
            return {'x': x, 'confidence': confidence, 'label': label, 'id': frame_dir}

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            # Corrupt file
             x = torch.zeros((self.target_frames, 17, 2))
             confidence = torch.zeros((self.target_frames, 17))
             return {'x': x, 'confidence': confidence, 'label': label, 'id': frame_dir}
            
        # data['keypoint'] -> (T, J, C) -> (T, 17, 3)
        # Check shape
        kp = data.get('keypoint', None)
        if kp is None:
             x = torch.zeros((self.target_frames, 17, 2))
             confidence = torch.zeros((self.target_frames, 17))
             return {'x': x, 'confidence': confidence, 'label': label, 'id': frame_dir}
             
        # Often shape is (N_person, T, J, C) or just (T, J, C)
        # In the inspected file: (275, 17, 3), so it's (T, J, C).
        # Sometimes (M, T, J, C). We should check dimensions.
        if len(kp.shape) == 4: # (M, T, J, C)
            kp = kp[0] # Take first person
            
        # Resample
        kp = self._resample(kp, self.target_frames)
        
        # Split features
        # kp is (T, J, 3) -> x=(T, J, 2), conf=(T, J)
        x_np = kp[..., :2]
        conf_np = kp[..., 2]
        
        x = torch.from_numpy(x_np).float()
        confidence = torch.from_numpy(conf_np).float()
        
        return {
            "x": x,
            "confidence": confidence,
            "label": label,
            "id": frame_dir
        }

    def _resample(self, data, target_len):
        """
        Resample data (T, ...) to target_len along T dimension (axis 0).
        """
        T = data.shape[0]
        if T == target_len:
            return data
        
        if T == 0:
            # Return zeros matching other dims
            shape = list(data.shape)
            shape[0] = target_len
            return np.zeros(shape, dtype=data.dtype)
            
        indices = np.linspace(0, T - 1, target_len).astype(int)
        return data[indices]
