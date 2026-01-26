import pickle
import numpy as np
import os

def inspect_kinetics():
    pkl_path = 'data/raw/skeleton/kinetics400/k400_2d.pkl'
    
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return

    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if 'split' in data:
            print(f"Split keys: {data['split'].keys()}")
            
        if 'annotations' in data:
            anns = data['annotations']
            print(f"Annotations type: {type(anns)}")
            print(f"Number of annotations: {len(anns)}")
            if isinstance(anns, list) and len(anns) > 0:
                print(f"Sample annotation: {anns[0]}")
            elif isinstance(anns, dict):
                 print(f"Annotations keys sample: {list(anns.keys())[:5]}")
                 
        if 'split' in data:
            splits = data['split']
            if 'train' in splits:
                train_split = splits['train']
                print(f"Train split type: {type(train_split)}")
                print(f"Train split sample (first 5): {train_split[:5]}")

                
    # Also check one .npy file
    # data/raw/skeleton/kinetics400/kpfiles/
    # The structure I saw earlier:
    # data/raw/skeleton/kinetics400/kpfiles/OpenMMLab___Kinetics400-skeleton/raw/k400_kpfiles_2d.zip
    # Wait, is it unzipped?
    # I saw data/raw/skeleton/kinetics400/kpfiles/OpenMMLab___Kinetics400-skeleton/raw/k400_kpfiles_2d.zip
    # The user request said: "k400_2d.pkl + kpfiles/*.npy"
    # "Dataset layout: data/raw/skeleton/kinetics400/kpfiles/{train,val,test}/*.npy"
    
    # I should check if the .npy files are actually extracted.
    kp_root = 'data/raw/skeleton/kinetics400/kpfiles'
    
    # List one file in kpfiles if possible
    print("\nChecking file structure...")
    for root, dirs, files in os.walk(kp_root):
        if len(files) > 0:
            npy_files = [f for f in files if f.endswith('.npy')]
            if npy_files:
                print(f"Found {len(npy_files)} .npy files in {root}")
                sample_npy = os.path.join(root, npy_files[0])
                print(f"Loading {sample_npy}...")
                npy_data = np.load(sample_npy, allow_pickle=True)
                print(f"NPY data shape: {npy_data.shape} type: {npy_data.dtype}")
                # Often it's a dict or structured array
                if npy_data.dtype == 'O': # Object
                    print(f"Content: {npy_data.item()}")
                break

if __name__ == "__main__":
    inspect_kinetics()
