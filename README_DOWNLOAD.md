# Download Instructions for Excluded Files

To reduce the size of this archive, large datasets and model checkpoints have been excluded.
Follow the instructions below to download or regenerate them.

## 1. Datasets (VizWiz)
The VizWiz dataset (images and annotations) is not included.
To download it, run:
```bash
python download_vizwiz.py
```
This will download the data into `data/vizwiz`.

## 2. Pre-trained Models (Depth Anything, Detectron2)
The backbone models required for feature extraction are not included.
To download them, run:
```bash
python download_checkpoints.py
```
This will download:
- `checkpoints/depth_anything_v2_vitl.pth`
- `output_X101/X-101.pth` (linked from a Detectron2 model)

## 3. Generated Checkpoints
The following training checkpoints were excluded:
- `saved_models/safe_dlct_vizwiz_last.pth`
- `saved_models/safe_dlct_vizwiz_xe_res.pth`

The **best** model checkpoint `saved_models/safe_dlct_vizwiz_best.pth` **IS INCLUDED** in this archive.
The others can be reproduced by running the training script:
```bash
bash run_training.sh
```

## 4. Dummy Data
If `dummy_features.h5` is missing and you need it for testing, you can generate it by running:
```bash
python generate_dummy_data.py
```
