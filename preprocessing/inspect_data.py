import pickle
import h5py
import os
import json
import numpy as np

def inspect_vocab():
    vocab_path = 'vocab_vizwiz.pkl'
    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} not found.")
        return

    print(f"--- Inspecting {vocab_path} ---")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Top 10 words: {vocab.itos[:10]}")
    
    # Check for specific VizWiz terms
    check_words = ['quality', 'expiry', 'text', 'monitor', 'screen', 'bottle', 'can']
    print("\nChecking for specific words:")
    for w in check_words:
        if w in vocab.stoi:
            print(f"  '{w}': index {vocab.stoi[w]}")
        else:
            print(f"  '{w}': NOT FOUND")

def inspect_features():
    feat_path = 'data/vizwiz/vizwiz_all.h5'
    if not os.path.exists(feat_path):
        print(f"Error: {feat_path} not found.")
        return

    print(f"\n--- Inspecting {feat_path} ---")
    try:
        with h5py.File(feat_path, 'r') as f:
            print("Keys:", list(f.keys()))
            if 'detections' in f:
                print("Detections shape:", f['detections'].shape)
                # Check for zero features
                sample = f['detections'][0]
                print(f"Sample feature mean: {np.mean(sample)}")
                print(f"Sample feature std: {np.std(sample)}")
                if np.all(sample == 0):
                    print("WARNING: Sample feature is all zeros!")
            else:
                # Might be organized differently, e.g. by image_id
                print("Structure seems different from standard 'detections' key.")
                keys = list(f.keys())[:5]
                print(f"First 5 keys: {keys}")
                first_key = keys[0]
                print(f"Shape of {first_key}: {f[first_key].shape}")

    except Exception as e:
        print(f"Error reading H5 file: {e}")

if __name__ == "__main__":
    inspect_vocab()
    # inspect_features()
