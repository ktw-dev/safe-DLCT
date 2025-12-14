import json
import numpy as np

def check_max_len(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    max_len = 0
    for img in data['images']:
        if 'processed_tokens' in img:
            for tokens in img['processed_tokens']:
                if len(tokens) > max_len:
                    max_len = len(tokens)
    
    print(f"Max caption length: {max_len}")
    return max_len

if __name__ == "__main__":
    check_max_len('data/vizwiz/dataset_vizwiz.json')
