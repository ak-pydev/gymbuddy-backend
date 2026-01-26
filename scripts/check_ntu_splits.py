import pickle
import sys

def check_splits():
    file_path = 'data/raw/skeleton/ntu120/ntu120_3d.pkl'
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    if 'split' in data:
        print(f"Split keys: {data['split'].keys()}")
        # Print first few items of one split
        key = list(data['split'].keys())[0]
        print(f"First 5 items in {key}: {data['split'][key][:5]}")
    else:
        print("No 'split' key found.")

if __name__ == "__main__":
    check_splits()
