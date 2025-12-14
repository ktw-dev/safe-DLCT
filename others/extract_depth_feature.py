import argparse
import os
import torch
import cv2
import numpy as np
import h5py
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from depth_anything_v2.dpt import DepthAnythingV2

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Depth Features using Depth Anything V2')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save HDF5 file')
    parser.add_argument('--box_path', type=str, default=None, help='Path to HDF5 file containing bounding boxes (optional)')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    
    image_files = [f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))]
    
    # Open box file if provided
    box_file = None
    if args.box_path:
        box_file = h5py.File(args.box_path, 'r')
        
    # Open in append mode to support resuming
    with h5py.File(args.save_path, 'a') as f:
        for img_name in tqdm(image_files):
            try:
                # Extract ID first to check if already processed
                try:
                    image_id = int(img_name.split('_')[-1].split('.')[0])
                except ValueError:
                    print(f"Warning: Could not extract ID from {img_name}, skipping.")
                    continue

                # Check if already processed
                if f'{image_id}_depth_grids' in f:
                    continue

                img_path = os.path.join(args.img_dir, img_name)
                raw_image = cv2.imread(img_path)
                
                if raw_image is None:
                    print(f"Warning: Could not read image {img_path}, skipping.")
                    continue
                    
                depth = model.infer_image(raw_image) # (H, W)
                
                # 1. Depth Grids (7x7 pooling)
                depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().to(device) # (1, 1, H, W)
                depth_grid = torch.nn.functional.adaptive_avg_pool2d(depth_tensor, (7, 7)) # (1, 1, 7, 7)
                depth_grid = depth_grid.view(49).cpu().numpy()
                
                f.create_dataset(f'{image_id}_depth_grids', data=depth_grid)
                
                # 2. Depth Regions (if boxes provided)
                if box_file:
                    box_key = f'{image_id}_boxes'
                    if box_key in box_file:
                        boxes = box_file[box_key][:] # (K, 4)
                        region_depths = []
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            # Clip coordinates
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(depth.shape[1], x2); y2 = min(depth.shape[0], y2)
                            
                            if x2 > x1 and y2 > y1:
                                region_depth = np.mean(depth[y1:y2, x1:x2])
                            else:
                                region_depth = 0.0
                            region_depths.append(region_depth)
                        
                        f.create_dataset(f'{image_id}_depth_regions', data=np.array(region_depths))
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    
    if box_file:
        box_file.close()

if __name__ == '__main__':
    main()
