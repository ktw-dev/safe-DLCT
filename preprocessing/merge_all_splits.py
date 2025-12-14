import h5py
import numpy as np
import os
from tqdm import tqdm

def merge_all_splits(base_dir, output_path):
    splits = ['train', 'val', 'test']
    
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Deleting...")
        os.remove(output_path)
        
    print(f"Creating {output_path}...")
    
    with h5py.File(output_path, 'w') as f_out:
        for split in splits:
            region_path = os.path.join(base_dir, f"vizwiz_{split}_region.hdf5")
            depth_path = os.path.join(base_dir, f"vizwiz_{split}_depth.h5")
            
            print(f"Processing {split} split...")
            print(f" - Region: {region_path}")
            print(f" - Depth:  {depth_path}")
            
            if not os.path.exists(region_path):
                print(f"   ⚠️  Region file not found for {split}. Skipping.")
                continue
            if not os.path.exists(depth_path):
                print(f"   ⚠️  Depth file not found for {split}. Skipping.")
                continue
                
            try:
                with h5py.File(region_path, 'r') as f_reg, h5py.File(depth_path, 'r') as f_dep:
                    # Get common IDs
                    reg_ids = set([k.split('_')[0] for k in f_reg.keys() if k.endswith('_features')])
                    dep_ids = set([k.split('_')[0] for k in f_dep.keys() if k.endswith('_depth_grids')])
                    
                    common_ids = sorted(list(reg_ids.intersection(dep_ids)))
                    print(f"   Found {len(common_ids)} common images in {split}.")
                    
                    for img_id in tqdm(common_ids, desc=f"Merging {split}"):
                        # Define keys to copy
                        keys_to_copy = [
                            (f_reg, f'{img_id}_features'),
                            (f_reg, f'{img_id}_boxes'),
                            (f_reg, f'{img_id}_classes'),
                            (f_reg, f'{img_id}_size'),
                            (f_reg, f'{img_id}_grids'),
                            (f_dep, f'{img_id}_depth_grids'),
                            (f_dep, f'{img_id}_depth_regions')
                        ]
                        
                        for source_file, key in keys_to_copy:
                            if key in source_file:
                                if key in f_out:
                                    # Duplicate check
                                    # print(f"Warning: Key {key} already exists. Overwriting.")
                                    del f_out[key]
                                
                                f_out.create_dataset(key, data=source_file[key][()])
                            else:
                                # Optional: Warn if expected key is missing (e.g. size might be missing in some datasets?)
                                # But for our pipeline, we expect them.
                                # print(f"Warning: Key {key} missing for {img_id}")
                                pass
                        
                        # Generate and save mask (Added for Safe-DLCT)
                        # Mask shape: (N_regions, N_grids)
                        # We need to get N_regions and N_grids from the data we just copied (or from source)
                        try:
                            feat_data = f_reg[f'{img_id}_features'][()]
                            grid_data = f_reg[f'{img_id}_grids'][()] # We know this exists now
                            
                            n_regions = feat_data.shape[0]
                            n_grids = grid_data.shape[0]
                            
                            # Create default mask (all ones, fully connected)
                            mask = np.ones((n_regions, n_grids), dtype='float32')
                            
                            mask_key = f'{img_id}_mask'
                            if mask_key in f_out:
                                del f_out[mask_key]
                            f_out.create_dataset(mask_key, data=mask)
                            
                        except Exception as e:
                            print(f"   ⚠️  Error generating mask for {img_id}: {e}")
                                
            except Exception as e:
                print(f"   ❌ Error processing {split}: {e}")

    print(f"\nSuccessfully created {output_path}")
    
    # Verification
    with h5py.File(output_path, 'r') as f:
        print(f"Total keys in merged file: {len(f.keys())}")

if __name__ == "__main__":
    base_dir = "data/vizwiz"
    output_path = os.path.join(base_dir, "vizwiz_all.h5")
    merge_all_splits(base_dir, output_path)
