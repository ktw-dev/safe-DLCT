# Safe-DLCT Project Summary

This document provides a comprehensive overview of the **Safe-DLCT** project, detailing the objectives, implemented features, completed tasks, debugging history, and current status.

## 1. Project Overview
**Safe-DLCT** is an extension of the DLCT (Dual-Level Collaborative Transformer) image captioning model, designed specifically for visually impaired users. It introduces safety-aware mechanisms to prioritize hazardous objects and ensure geometrically consistent captions.

### Core Features Implemented
*   **D-HAG (Depth-Aware Hazard Alignment Graph)**: Modifies the attention mechanism to consider depth differences, ensuring the model understands spatial relationships better.
*   **Hazard-Weighted LCCA**: Biases the attention mechanism towards hazardous objects (e.g., Person, Car, Bicycle) to ensure they are mentioned in the caption.
*   **Geometry-Consistent Reward**: A reinforcement learning (SCST) reward component that penalizes captions describing physically impossible spatial relationships (e.g., "person in front of car" when depth says otherwise).

---

## 2. Completed Tasks
The following tasks have been successfully completed:

### ✅ Safe-DLCT Core Implementation
- [x] **D-HAG Implementation**: Modified `data/field.py` to load depth data and prune the alignment graph based on depth thresholds.
- [x] **Hazard Attention**: Modified `models/DLCT/attention.py` to inject hazard-weighted bias into the attention scores.
- [x] **Geometry Reward**: Implemented `utils/geo_reward.py` to parse spatial relations and compute consistency rewards.
- [x] **Training Pipeline**: Updated `train.py` to integrate the new reward into the SCST loop.
- [x] **Dry Run Verification**: Successfully ran a full training cycle (XE + SCST) with dummy data to verify pipeline stability.

### ✅ VizWiz Dataset Integration
- [x] **Data Download**: Created `download_vizwiz.py` to fetch the official VizWiz dataset.
- [x] **Label Preprocessing**: Created `prepro_labels_vizwiz.py` to convert VizWiz annotations to Karpathy JSON format, filtering out "precanned" and "rejected" captions.
- [x] **Feature Extraction Setup**:
    - Refactored `extract_region_feature.py` to use a **Visual Genome pre-trained ResNeXt-101** model.
    - Implemented **Resume Capability** to handle interruptions during long extraction processes.
    - Resolved critical **Model Architecture Mismatches** (FPN vs. C4, Group settings).

---

## 3. Walkthrough & Debugging Log

### Phase 1: Dry Run & Pipeline Stabilization
*   **Issue**: `spacy` model loading error (`en` vs `en_core_web_sm`).
    *   **Fix**: Updated model name in `data/utils.py`.
*   **Issue**: Beam search integer division error causing `gather` failure.
    *   **Fix**: Corrected division operator in `models/beam_search/beam_search.py`.
*   **Outcome**: Successfully trained a model on dummy data for 1 epoch, verifying the code logic.

### Phase 2: Feature Extraction (The "Incompatible Shapes" Saga)
*   **Challenge**: Loading Visual Genome weights (`X-101.pth`) into Detectron2.
*   **Issue 1**: "Skip loading parameter... incompatible shapes".
    *   **Cause**: Mismatch between the checkpoint's **C4 architecture** (Res5 in backbone) and the config's **FPN architecture**.
    *   **Fix**: Created `configs/VG-X101-grid.yaml` using `StandardROIHeads` with `res5` input and 1x1 pooling to match the checkpoint structure perfectly.
*   **Issue 2**: `RuntimeError: Given groups=1, expected input...`.
    *   **Cause**: The checkpoint used **ResNeXt-101 (32x8d)** but the default config used ResNet-101.
    *   **Fix**: Updated config to set `NUM_GROUPS: 32` and `WIDTH_PER_GROUP: 8`.
*   **Issue 3**: Silent Failures & Interruption.
    *   **Fix**: Added `python -u` for unbuffered logging and implemented **Resume Logic** (append mode) in `extract_region_feature.py`.

---

## 4. Current Status
*   **Feature Extraction**: Currently running on the `train` split.
    *   **Progress**: ~91% completed (21,870 / 23,954 images).
    *   **Status**: Process has been restarted with resume logic to finish the remaining images.
*   **Codebase**: Fully compressed into `safe-DLCT_code_only.tar.gz` (92MB) for easy transfer.

---

## 5. Deliverables & Resources
To continue this project in a new environment, you need the following:

### 📦 Code Archive
*   **File**: `safe-DLCT_code_only.tar.gz` (Located in project root)
*   **Contents**: All source code, configs, and scripts. Excludes large data and weights.

### 🔗 External Resources (Download Links)
If you move to a new environment, download these files again:

1.  **Visual Genome Weights (X-101)**
    *   URL: `https://dl.fbaipublicfiles.com/grid-feats-vqa/X-101/X-101_32x8d_FPN_3x.pth`
2.  **Depth Anything V2 Weights**
    *   URL: `https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth`
3.  **VizWiz Dataset**
    *   Train Images: `https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip`
    *   Val Images: `https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip`
    *   Test Images: `https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip`
    *   Annotations: `https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip`

---

## 6. How to Resume Work
1.  Unzip `safe-DLCT_code_only.tar.gz`.
2.  Download the **VizWiz Dataset** into `data/vizwiz`.
3.  Download the **Model Weights** (links above).
4.  Run `bash run_all_extraction.sh` to finish feature extraction (it will automatically resume).
5.  Run `python prepare_data.py` to merge features.
6.  Run `python train.py --dataset vizwiz ...` to start training.
