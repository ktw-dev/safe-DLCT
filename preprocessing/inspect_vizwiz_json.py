import json
import os

ann_file = 'data/vizwiz/dataset_vizwiz.json'

if not os.path.exists(ann_file):
    print(f"File not found: {ann_file}")
    exit(1)

with open(ann_file, 'r') as f:
    data = json.load(f)

print(f"Total images: {len(data['images'])}")
splits = {}
for img in data['images']:
    s = img.get('split', 'unknown')
    splits[s] = splits.get(s, 0) + 1

print("Split counts:", splits)

test_with_caps = 0
test_without_caps = 0

for img in data['images']:
    if img.get('split') == 'test':
        if 'captions' in img and len(img['captions']) > 0:
            test_with_caps += 1
        else:
            test_without_caps += 1

print(f"Test images with captions: {test_with_caps}")
print(f"Test images without captions: {test_without_caps}")
