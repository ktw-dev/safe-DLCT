import h5py
import numpy as np
import os

def repair_missing_depth(region_path, depth_path, missing_ids):
    print(f"Starting repair for: {missing_ids}")
    
    if not os.path.exists(region_path) or not os.path.exists(depth_path):
        print("Error: Files not found.")
        return

    # r+: 읽기/쓰기 모드 (기존 데이터 유지)
    with h5py.File(region_path, 'r') as f_reg, h5py.File(depth_path, 'r+') as f_depth:
        for img_id in missing_ids:
            try:
                print(f"Repairing ID: {img_id}...")
                
                # 1. Region 파일에서 박스 개수 파악 (Shape을 맞춰야 함)
                box_key = f'{img_id}_boxes'
                if box_key in f_reg:
                    num_boxes = f_reg[box_key].shape[0]
                else:
                    print(f"  Warning: ID {img_id} not found in Region file either.")
                    continue

                # 2. Depth Grids 복구 (항상 49개)
                grid_key = f'{img_id}_depth_grids'
                if grid_key not in f_depth:
                    # 7x7 = 49
                    f_depth.create_dataset(grid_key, data=np.zeros((49,), dtype=np.float32))
                    print(f"  -> Created {grid_key} (Zeros)")
                else:
                    print(f"  -> {grid_key} already exists.")

                # 3. Depth Regions 복구 (박스 개수만큼)
                region_depth_key = f'{img_id}_depth_regions'
                if region_depth_key not in f_depth:
                    f_depth.create_dataset(region_depth_key, data=np.zeros((num_boxes,), dtype=np.float32))
                    print(f"  -> Created {region_depth_key} (Zeros, Size: {num_boxes})")
                else:
                    print(f"  -> {region_depth_key} already exists.")
                    
            except Exception as e:
                print(f"  Failed to repair {img_id}: {e}")

    print("Repair completed.")

if __name__ == "__main__":
    # 경로 확인 필수
    base_dir = "data/vizwiz"
    r_path = os.path.join(base_dir, "vizwiz_train_region.hdf5")
    d_path = os.path.join(base_dir, "vizwiz_train_depth.h5")
    
    # 아까 integrity 체크에서 나온 누락 ID들
    missing_list = ['11477', '18769'] 
    
    repair_missing_depth(r_path, d_path, missing_list)