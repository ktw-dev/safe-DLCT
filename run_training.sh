#!/bin/bash

# Activate conda environment if needed (optional, assuming user runs this in correct env)
# source activate dlct

echo "Starting Safe-DLCT Training on VizWiz..."

/home/user/miniconda3/envs/dlct/bin/python train.py \
    --dataset vizwiz \
    --features_path data/vizwiz/vizwiz_all.h5 \
    --resume_best \
    --annotation_folder data/vizwiz \
    --exp_name safe_dlct_vizwiz_v2 \
    --image_field ImageAllFieldWithMask \
    --model DLCT \
    --dim_feats 2048 \
    --batch_size 32 \
    --workers 4 \
    --head 8 \
    --d_model 512 \
    --n_layer 3 \
    --rl_at 30 \
    --logs_folder tensorboard_logs
