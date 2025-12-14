import torch
import sys

def check_shape(path):
    try:
        data = torch.load(path, map_location='cpu')
        if 'model' in data:
            st = data['model']
        elif 'state_dict' in data:
            st = data['state_dict']
        else:
            st = data
            
        key = 'roi_heads.box_head.fc1.weight'
        if key in st:
            print(f"{key}: {st[key].shape}")
        else:
            print(f"{key} not found")
            
        key2 = 'roi_heads.box_predictor.cls_score.weight'
        if key2 in st:
            print(f"{key2}: {st[key2].shape}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_shape(sys.argv[1])
