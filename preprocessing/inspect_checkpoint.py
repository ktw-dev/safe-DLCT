import torch
import sys

def inspect_checkpoint(path):
    try:
        data = torch.load(path, map_location='cpu')
        if 'model' in data:
            keys = list(data['model'].keys())
        elif 'state_dict' in data:
            keys = list(data['state_dict'].keys())
        else:
            keys = list(data.keys())
        
        print(f"Total keys: {len(keys)}")
        for k in keys:
            print(k)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        print("Usage: python inspect_checkpoint.py <path>")
