import argparse
import os
import glob
import h5py
import numpy as np
import torch
import tqdm
import pickle
import cv2
import sys

# Monkey patch torch.load to disable weights_only by default and fallback to pickle
_original_load = torch.load
def _safe_load(f, map_location=None, pickle_module=None, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    try:
        return _original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    except RuntimeError as e:
        if "Invalid magic number" in str(e):
            # Fallback to standard pickle
            if isinstance(f, str):
                with open(f, 'rb') as stream:
                    return pickle.load(stream)
            else:
                return pickle.load(f)
        raise e
torch.load = _safe_load

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.data.detection_utils import read_image
from detectron2.modeling.roi_heads import Res5ROIHeads, StandardROIHeads

# Add path to find others.grid_feats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from others.grid_feats import add_attribute_config

def setup(args):
    cfg = get_cfg()
    add_attribute_config(cfg)
    
    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"Warning: Weights not found at {args.weights}. Switching to standard R-101 C4 model.", flush=True)
        # Use standard R-101 C4 config from Detectron2 model zoo
        from detectron2 import model_zoo
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
    else:
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.MODEL.WEIGHTS = args.weights
        
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.freeze()
    return cfg



def get_image_files(img_dir):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
        files.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    return sorted(list(set(files)))

def main(args):
    print("Entering main...", flush=True)
    cfg = setup(args)
    print("Setup complete.", flush=True)
    # Load model
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    print("Weights loaded.")
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    print("Model to device complete.", flush=True)

    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images in {args.image_dir}", flush=True)

    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # Open in append mode to support resuming
    f = h5py.File(args.output_file, 'a')
    print(f"Opened {args.output_file} in append mode.", flush=True)
    
    # Define transform for resizing
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    max_regions = 100
    
    print("Starting inference loop...", flush=True)
    for i, img_path in tqdm.tqdm(enumerate(image_files), total=len(image_files)):
        basename = os.path.basename(img_path)
        try:
            # VizWiz_train_00000000.jpg -> 0
            img_id = int(basename.split('_')[-1].split('.')[0])
            
            # Check if already processed
            if f'{img_id}_features' in f:
                continue
                
        except ValueError:
            # Fallback or skip
            print(f"Warning: Could not extract ID from {basename}, skipping.")
            continue

        try:
            original_image = read_image(img_path, format="BGR")
            height, width = original_image.shape[:2]
            
            # Apply preprocessing
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = [{"image": image, "height": height, "width": width}]
            
            with torch.no_grad():
                images = model.preprocess_image(inputs)
                features = model.backbone(images.tensor)
                proposals, _ = model.proposal_generator(images, features)
                
                # Extract features

                # Extract features
                if isinstance(model.roi_heads, Res5ROIHeads):
                    # C4 Model Logic
                    features_list = [features[f] for f in model.roi_heads.in_features]
                    # _shared_roi_transform calls pooler + res5
                    box_features = model.roi_heads._shared_roi_transform(features_list, [x.proposal_boxes for x in proposals])
                    # box_features is [N, 2048, 7, 7], we need [N, 2048]
                    box_features = box_features.mean(dim=[2, 3])
                else:
                    # Standard/FPN Model Logic
                    features_list = [features[f] for f in model.roi_heads.in_features]
                    box_features1 = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
                    box_features = model.roi_heads.box_head(box_features1)
                
                predictions = model.roi_heads.box_predictor(box_features)
                pred_instances, index = model.roi_heads.box_predictor.inference(predictions, proposals)
                
                # Select top-k regions
                topk = 10
                if len(pred_instances) > 0:
                    scores = pred_instances[0].get_fields()['scores']
                    topk_index = index[0][:topk]
                    
                    thresh_mask = scores > cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
                    thresh_index = index[0][thresh_mask]
                    
                    if len(thresh_index) < 10:
                        final_index = topk_index
                    elif len(thresh_index) > max_regions:
                        final_index = thresh_index[:max_regions]
                    else:
                        final_index = thresh_index
                    

                if len(pred_instances) > 0:
                    final_boxes = pred_instances[0].pred_boxes
                    final_scores = pred_instances[0].scores
                    final_classes = pred_instances[0].pred_classes

                    # 1. Score Threshold Filtering
                    keep = final_scores > cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
                    
                    # 2. Top-K / Min-K Logic (Count correction)
                    num_keep = keep.sum().item()
                    if num_keep < 10:
                        # Keep at least 10 (even if score is low)
                        k = min(10, len(final_scores))
                        _, topk_indices = final_scores.topk(k)
                        keep = torch.zeros_like(final_scores, dtype=torch.bool)
                        keep[topk_indices] = True
                    elif num_keep > max_regions:
                        # Limit max regions (usually 100)
                        _, topk_indices = final_scores.topk(max_regions)
                        keep = torch.zeros_like(final_scores, dtype=torch.bool)
                        keep[topk_indices] = True
                    
                    # 3. Apply Filtering (Boxes & Classes)
                    selected_boxes = final_boxes[keep]
                    selected_classes = final_classes[keep]
                    
                    # 4. Feature Mapping (Key Fix)
                    # Find original proposal indices using keep mask
                    keep_indices = torch.nonzero(keep).squeeze(1)
                    original_indices = index[0][keep_indices]
                    
                    # Get directly from box_features (2048-d) (No mean operation!)
                    selected_feat = box_features[original_indices]
                    selected_boxes_tensor = selected_boxes.tensor
                    
                    # Debug print for the first image
                    if i == 0:
                        print(f"Debug [ID: {img_id}]:", flush=True)
                        print(f" - Feature Shape: {selected_feat.shape} (Must be [N, 2048])", flush=True)
                        print(f" - Box Shape: {selected_boxes_tensor.shape}", flush=True)

                    # 5. Grid Features Extraction (Added for Safe-DLCT)
                    if isinstance(model.roi_heads, Res5ROIHeads):
                        # For C4, we take the res4 feature map and pass it through res5
                        # features is a dict, usually {'res4': ...}
                        feature_map = features[model.roi_heads.in_features[0]] # (1, 1024, H, W)
                        grid_feat = model.roi_heads.res5(feature_map) # (1, 2048, H, W)
                        
                        # Adaptive Average Pooling to 7x7
                        grid_feat = torch.nn.functional.adaptive_avg_pool2d(grid_feat, (7, 7)) # (1, 2048, 7, 7)
                        
                        # Reshape to (49, 2048)
                        grid_feat = grid_feat.permute(0, 2, 3, 1).reshape(49, 2048) # (49, 2048)
                    else:
                        # For FPN/Standard, we might use the highest level feature P5
                        # But typically this script is used with C4 for captioning.
                        # Fallback: Use the last feature map and pool
                        feat_name = model.roi_heads.in_features[-1]
                        feature_map = features[feat_name]
                        grid_feat = torch.nn.functional.adaptive_avg_pool2d(feature_map, (7, 7))
                        # If channels are not 2048, this might be an issue, but let's assume C4 mostly.
                        # FPN P5 is usually 256d. This would require a projection.
                        # Given the user is using VG-X101-grid.yaml (C4), the C4 block above should trigger.
                        grid_feat = grid_feat.permute(0, 2, 3, 1).reshape(49, -1)

                    f.create_dataset(f'{img_id}_features', data=selected_feat.cpu().numpy())
                    f.create_dataset(f'{img_id}_boxes', data=selected_boxes_tensor.cpu().numpy())
                    f.create_dataset(f'{img_id}_classes', data=selected_classes.cpu().numpy())
                    f.create_dataset(f'{img_id}_size', data=np.array([[height, width]]))
                    f.create_dataset(f'{img_id}_grids', data=grid_feat.cpu().numpy())
                    
                else:
                    # No detections Fallback (1024-d dummy data)
                    print(f"No detections for {basename}", flush=True)
                    f.create_dataset(f'{img_id}_features', data=np.zeros((1, 1024), dtype='float32'))
                    f.create_dataset(f'{img_id}_boxes', data=np.zeros((1, 4), dtype='float32'))
                    f.create_dataset(f'{img_id}_classes', data=np.zeros((1,), dtype='int64'))
                    f.create_dataset(f'{img_id}_size', data=np.array([[height, width]]))
                    f.create_dataset(f'{img_id}_grids', data=np.zeros((49, 2048), dtype='float32'))

        except Exception as e:
            print(f"Error processing {img_path}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    f.close()
    print(f"Saved features to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_file", required=True, help="Path to save HDF5 file")
    parser.add_argument("--config-file", default="configs/X-101-grid.yaml", help="Path to config file")
    parser.add_argument("--weights", default="data/genome/1600-400-20/model_final.pth", help="Path to model weights")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Modify config options")
    args = parser.parse_args()
    main(args)
