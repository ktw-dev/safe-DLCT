import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.DLCT.attention import ScaledDotProductWithBoxAttention
from utils.geo_reward import get_geo_reward

def test_hazard_bias():
    print("Testing Hazard Bias...")
    att = ScaledDotProductWithBoxAttention(d_model=64, d_k=64, d_v=64, h=1)
    q = torch.randn(1, 1, 64)
    k = torch.randn(1, 2, 64)
    v = torch.randn(1, 2, 64)
    
    # Case 1: No hazard
    box_relation = torch.randn(1, 1, 1, 2)
    out1 = att(q, k, v, box_relation_embed_matrix=box_relation)
    
    # Case 2: Hazard mask (2nd key is hazard)
    # hazard_mask shape: (b_s, 1, 1, nk)
    hazard_mask = torch.tensor([[[[0.0, 1.0]]]]) 
    out2 = att(q, k, v, box_relation_embed_matrix=box_relation, hazard_mask=hazard_mask)
    
    diff = torch.abs(out1 - out2).sum()
    print(f"Difference with hazard mask: {diff.item()}")
    
    if diff.item() > 0:
        print("Hazard Bias Test Passed!")
    else:
        print("Hazard Bias Test Failed: No difference in output")
        sys.exit(1)

def test_geo_reward():
    print("Testing Geo Reward...")
    # Mock data
    caps_gen = ["person in front of car", "car behind person", "cat on mat"]
    
    # Depths: smaller means closer
    depth_regions = np.array([
        [0.5, 0.8, 0.0], # person(idx 0)=0.5, car(idx 1)=0.8. Person < Car. Person is closer.
        [0.8, 0.5, 0.0], # car(idx 0)=0.8, person(idx 1)=0.5. Person < Car.
        [0.5, 0.5, 0.0]
    ])
    
    # Classes: Person=1, Car=3
    classes = np.array([
        [1, 3, 0], # person, car
        [3, 1, 0], # car, person
        [16, 17, 0] # cat, mat
    ])
    
    rewards = get_geo_reward(caps_gen, depth_regions, classes)
    print(f"Rewards: {rewards}")
    
    # Expected:
    # 1. "person in front of car": Person(0.5) < Car(0.8). Correct. Reward +1.0.
    # 2. "car behind person": Not handled by my simple logic (only checks "front"). Returns 0.0.
    # 3. "cat on mat": Not handled. Returns 0.0.
    
    if rewards[0] == 1.0:
        print("Geo Reward Test Passed for Case 1!")
    else:
        print(f"Geo Reward Test Failed for Case 1: Expected 1.0, got {rewards[0]}")
        sys.exit(1)

if __name__ == "__main__":
    test_hazard_bias()
    test_geo_reward()
