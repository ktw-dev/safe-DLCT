import h5py
import os
import sys

def check_grids(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Checking {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            grid_keys = [k for k in keys if k.endswith('_grids')]
            feature_keys = [k for k in keys if k.endswith('_features')]
            
            print(f"  Total keys: {len(keys)}")
            print(f"  Feature keys: {len(feature_keys)}")
            print(f"  Grid keys: {len(grid_keys)}")
            
            if len(grid_keys) == 0:
                print("  ❌ NO '_grids' keys found!")
            elif len(grid_keys) < len(feature_keys):
                print(f"  ⚠️  Mismatch: {len(feature_keys)} features but only {len(grid_keys)} grids.")
            else:
                print("  ✅ Grids seem present.")
                # Check first one
                first_grid = grid_keys[0]
                print(f"  Sample {first_grid} shape: {f[first_grid].shape}")

    except Exception as e:
        print(f"  Error reading file: {e}")

if __name__ == "__main__":
    files = [
        "data/vizwiz/vizwiz_train_region.hdf5",
        "data/vizwiz/vizwiz_val_region.hdf5",
        "data/vizwiz/vizwiz_test_region.hdf5",
        "data/vizwiz/vizwiz_all.h5"
    ]
    
    for p in files:
        check_grids(p)
        print("-" * 30)
