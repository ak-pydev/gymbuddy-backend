import pickle
import numpy as np
import os
import sys

def inspect_ntu120():
    file_path = 'data/raw/skeleton/ntu120/ntu120_3d.pkl'
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    # Determine structure
    print(f"Data type: {type(data)}")
    
    samples = []
    labels = []
    
    
    if isinstance(data, list):
        print(f"Number of samples: {len(data)}")
        if len(data) > 0:
            sample = data[0]
            print(f"Sample type: {type(sample)}")
            if isinstance(sample, dict):
                print(f"Sample keys: {sample.keys()}")
                
                # Try to identify skeleton field
                skeleton_keys = ['keypoint', 'frame', 'skeleton', 'data']
                skel_key = next((k for k in skeleton_keys if k in sample), None)
                
                if skel_key:
                    skel_data = sample[skel_key]
                    print(f"One example's shape (T, J, D): {skel_data.shape}")
                else:
                    # Fallback or specific known keys for NTU
                    # Sometimes it's 'original_shape', 'img_shape', 'keypoint'
                    print("Could not automatically identify skeleton tensor. Printing sample content summary.")
                    for k, v in sample.items():
                        if hasattr(v, 'shape'):
                            print(f"  {k}: shape={v.shape}")
                        else:
                            print(f"  {k}: {v}")

                # Check labels
                label_keys = ['label', 'frame_label', 'type']
                lbl_key = next((k for k in label_keys if k in sample), None)
                
                if lbl_key:
                    labels = [item[lbl_key] for item in data]
                    unique_labels = set(labels)
                    print(f"Number of classes: {len(unique_labels)}")
                    
                    # Confirm labels are integers
                    all_ints = all(isinstance(l, (int, np.integer)) for l in labels)
                    print(f"Confirms labels are integers: {all_ints}")
                    if not all_ints:
                        print(f"Sample labels: {labels[:5]}")
                else:
                     print("Could not find label key.")
            else:
                 print("Data items are not dicts.")
    elif isinstance(data, dict):
        print(f"Keys in data: {data.keys()}")
        if 'annotations' in data:
            anns = data['annotations']
            print(f"Number of samples (annotations): {len(anns)}")
            
            if len(anns) > 0:
                sample = anns[0]
                print(f"Sample type: {type(sample)}")
                if isinstance(sample, dict):
                    print(f"Sample keys: {sample.keys()}")
                    
                    # Check for keypoint/skeleton
                    skeleton_keys = ['keypoint', 'frame', 'skeleton', 'data', 'keypoint_score', 'total_frames']
                    for k in sample:
                        if k in skeleton_keys or hasattr(sample[k], 'shape'):
                            val = sample[k]
                            shape = val.shape if hasattr(val, 'shape') else "No shape"
                            print(f"  {k}: {shape}")

                    # One example's shape (T, J, D)
                    # NTU usually has 'keypoint' with shape (1, T, J, 3) or (M, T, J, 3) where M is num_persons
                    if 'keypoint' in sample:
                        print(f"One example's 'keypoint' shape: {sample['keypoint'].shape}")

                    # Check labels
                    if 'label' in sample:
                        labels = [item['label'] for item in anns]
                        unique_labels = set(labels)
                        print(f"Number of classes: {len(unique_labels)}")
                        all_ints = all(isinstance(l, (int, np.integer)) for l in labels)
                        print(f"Confirms labels are integers: {all_ints}")
                    else:
                        print("Could not find 'label' key in annotations.")
                else:
                    print(f"Annotation item is not a dict: {sample}")

    else:
        print("Unknown data structure.")

if __name__ == "__main__":
    inspect_ntu120()
