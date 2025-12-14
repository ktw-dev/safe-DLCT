import h5py
import numpy as np
import os
import argparse

def check_split(split_name, region_path, depth_path):
    print(f"==================================================")
    print(f"Checking [{split_name}] Split Integrity...")
    print(f" - Region File: {region_path}")
    print(f" - Depth File:  {depth_path}")
    
    if not os.path.exists(region_path) or not os.path.exists(depth_path):
        print("❌ Error: One of the files does not exist.")
        return

    try:
        f_region = h5py.File(region_path, 'r')
        f_depth = h5py.File(depth_path, 'r')
    except Exception as e:
        print(f"❌ Error opening files: {e}")
        return

    # 1. 키(Key) 수집 및 ID 추출
    # Region은 '{id}_features' 형태, Depth는 '{id}_depth_grids' 형태라고 가정
    region_ids = set()
    for k in f_region.keys():
        if k.endswith('_features'):
            region_ids.add(k.split('_')[0])
            
    depth_ids = set()
    for k in f_depth.keys():
        if k.endswith('_depth_grids'):
            depth_ids.add(k.split('_')[0])

    print(f" - Total Region IDs: {len(region_ids)}")
    print(f" - Total Depth IDs:  {len(depth_ids)}")

    # 2. 누락 확인 (Region에는 있는데 Depth에 없는 것)
    missing_in_depth = region_ids - depth_ids
    missing_in_region = depth_ids - region_ids

    if len(missing_in_depth) > 0:
        print(f"🚨 [CRITICAL] Found {len(missing_in_depth)} images missing in Depth file!")
        print(f"   Example missing IDs: {list(missing_in_depth)[:5]} ...")
    else:
        print(f"✅ Depth file has all keys corresponding to Region file.")

    if len(missing_in_region) > 0:
        print(f"⚠️  [WARNING] Found {len(missing_in_region)} images in Depth but not in Region (Unlikely but check).")
    
    # 3. 데이터 검증 (0으로 채워진 더미인지, 실제 데이터인지 샘플링 확인)
    print(" - Verifying data content (Sampling 5 entries)...")
    sample_ids = list(depth_ids)[:5]
    for i, img_id in enumerate(sample_ids):
        grid_key = f"{img_id}_depth_grids"
        data = f_depth[grid_key][:]
        
        # 데이터가 모두 0인지 확인 (더미 데이터 체크)
        if np.all(data == 0):
            print(f"   ⚠️  Warning: ID {img_id} seems to be DUMMY data (all zeros).")
        else:
            if i == 0:
                print(f"   ℹ️  ID {img_id} looks valid. Shape: {data.shape}, Mean: {np.mean(data):.4f}")

    f_region.close()
    f_depth.close()
    print(f"Done.")
    print(f"==================================================\n")

if __name__ == "__main__":
    # 경로를 본인 환경에 맞게 수정하세요
    base_dir = "data/vizwiz"
    
    # Train 확인
    check_split("train", 
                os.path.join(base_dir, "vizwiz_train_region.hdf5"), 
                os.path.join(base_dir, "vizwiz_train_depth.h5"))
    
    # Val 확인 (필요시 주석 해제)
    check_split("val", 
                os.path.join(base_dir, "vizwiz_val_region.hdf5"), 
                os.path.join(base_dir, "vizwiz_val_depth.h5"))
    
    # Test 확인 (필요시 주석 해제)
    check_split("test", 
                os.path.join(base_dir, "vizwiz_test_region.hdf5"), 
                os.path.join(base_dir, "vizwiz_test_depth.h5"))