# Reconstructed grid_feats.py based on usage in extract_region_feature.py
from detectron2.config import CfgNode as CN
from detectron2.data import build_detection_test_loader

def add_attribute_config(cfg):
    """
    Add config for attribute prediction.
    """
    cfg.MODEL.ROI_BOX_HEAD.ATTR_NUM_CLASSES = 400
    cfg.MODEL.ROI_BOX_HEAD.ATTR_LOSS_WEIGHT = 0.1
    # Add other potential configs used in grid-feats-vqa
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

def build_detection_test_loader_with_attributes(cfg, dataset_name):
    """
    Build a test loader that includes attributes.
    For now, we use the default loader which should be sufficient for feature extraction
    if the dataset is registered correctly.
    """
    return build_detection_test_loader(cfg, dataset_name)
