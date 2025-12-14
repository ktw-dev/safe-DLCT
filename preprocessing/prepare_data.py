import h5py
import numpy as np
import argparse
from tqdm import tqdm
import os

def merge_data(region_path, depth_path, output_path):
    print(f"Merging {region_path} and {depth_path} into {output_path}...")
    
    if not os.path.exists(region_path):
        print(f"Region file {region_path} not found.")
        return
    if not os.path.exists(depth_path):
        print(f"Depth file {depth_path} not found.")
        return

    with h5py.File(region_path, 'r') as f_region, \
         h5py.File(depth_path, 'r') as f_depth, \
         h5py.File(output_path, 'w') as f_out:
        
        # Get intersection of image IDs
        region_keys = set([k.split('_')[0] for k in f_region.keys() if '_' in k])
        depth_keys = set([k.split('_')[0] for k in f_depth.keys() if '_' in k])
        
        common_ids = sorted(list(region_keys.intersection(depth_keys)))
        print(f"Found {len(common_ids)} common images.")
        
        for image_id in tqdm(common_ids):
            # Copy datasets
            try:
                # Region data
                f_out.create_dataset(f'{image_id}_features', data=f_region[f'{image_id}_features'][()])
                f_out.create_dataset(f'{image_id}_boxes', data=f_region[f'{image_id}_boxes'][()])
                f_out.create_dataset(f'{image_id}_classes', data=f_region[f'{image_id}_classes'][()])
                # Size might be useful
                if f'{image_id}_size' in f_region:
                     f_out.create_dataset(f'{image_id}_size', data=f_region[f'{image_id}_size'][()])

                # Depth data
                f_out.create_dataset(f'{image_id}_depth_regions', data=f_depth[f'{image_id}_depth_regions'][()])
                f_out.create_dataset(f'{image_id}_depth_grids', data=f_depth[f'{image_id}_depth_grids'][()])
                
            except KeyError as e:
                print(f"Error processing {image_id}: Missing key {e}")
                continue
            except Exception as e:
                print(f"Error processing {image_id}: {e}")
                continue

    print(f"Saved merged data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_file', type=str, required=True, help='Path to region HDF5 file')
    parser.add_argument('--depth_file', type=str, required=True, help='Path to depth HDF5 file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output HDF5 file')
    args = parser.parse_args()
    
    merge_data(args.region_file, args.depth_file, args.output_file)
