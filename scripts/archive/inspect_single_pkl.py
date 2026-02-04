import pickle
import numpy as np

def inspect_pkl():
    path = 'temp/W-qaXgFW70Y.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data type: {type(data)}")
    print(f"Data keys: {data.keys()}")
    # Check shape
    if 'keypoint' in data:
        print(f"Keypoint shape: {data['keypoint'].shape}")
    else:
        # Maybe it's not a dict?
        pass
    
    # Check structure
    for k, v in data.items():
        if hasattr(v, 'shape'):
             print(f"{k}: {v.shape}")
        else:
             print(f"{k}: {v}")

if __name__ == "__main__":
    inspect_pkl()
