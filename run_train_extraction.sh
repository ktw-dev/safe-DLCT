#!/bin/bash
set -e

# Function to run extraction
run_split() {
    SPLIT=$1
    IMG_DIR=$2
    
    REGION_FILE="data/vizwiz/vizwiz_${SPLIT}_region.hdf5"
    DEPTH_FILE="data/vizwiz/vizwiz_${SPLIT}_depth.h5"
    
    echo "=================================================="
    echo "Processing ${SPLIT} split"
    echo "Image Directory: ${IMG_DIR}"
    echo "Region File: ${REGION_FILE}"
    echo "Depth File: ${DEPTH_FILE}"
    echo "=================================================="
    
    echo "[${SPLIT}] Starting Region Extraction..."
    /home/user/miniconda3/envs/dlct/bin/python -u others/extract_region_feature.py \
        --image_dir ${IMG_DIR} \
        --output_file ${REGION_FILE} \
        --config-file "configs/VG-X101-grid.yaml"
        
    echo "[${SPLIT}] Starting Depth Extraction..."
    /home/user/miniconda3/envs/dlct/bin/python -u others/extract_depth_feature.py \
        --img_dir ${IMG_DIR} \
        --save_path ${DEPTH_FILE} \
        --box_path ${REGION_FILE} \
        --encoder vitl
        
    echo "[${SPLIT}] Completed."
}

# Run for all splits
run_split "train" "data/vizwiz/train"

echo "All extractions completed successfully."
