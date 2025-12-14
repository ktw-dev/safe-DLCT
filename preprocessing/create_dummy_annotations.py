import json
import os
import numpy as np

def create_dummy_annotations(output_dir='annotations'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Image IDs must match dummy_features.h5 (1000 to 1004)
    image_ids = [1000, 1001, 1002, 1003, 1004]
    
    dataset = {
        "images": [],
        "annotations": [],
        "type": "captions",
        "info": "dummy",
        "licenses": "dummy"
    }
    
    ann_id = 0
    for img_id in image_ids:
        dataset["images"].append({
            "id": img_id,
            "width": 640,
            "height": 480,
            "file_name": f"COCO_val2014_{img_id:012d}.jpg"
        })
        
        # Add 5 captions per image
        captions = [
            "person in front of car",
            "cat on the mat",
            "dog running in park",
            "bird flying in sky",
            "man holding a bat"
        ]
        
        for cap in captions:
            dataset["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "caption": cap
            })
            ann_id += 1
            
    # Save as train and val
    with open(os.path.join(output_dir, 'captions_train2014.json'), 'w') as f:
        json.dump(dataset, f)
        
    with open(os.path.join(output_dir, 'captions_val2014.json'), 'w') as f:
        json.dump(dataset, f)
        
    # Create dummy ID files expected by dataset.py
    # dataset.py expects ANNOTATION IDs, not Image IDs.
    # We have created annotations with IDs 0 to ann_id-1.
    all_ann_ids = np.arange(ann_id)
    
    np.save(os.path.join(output_dir, 'coco_train_ids.npy'), all_ann_ids)
    np.save(os.path.join(output_dir, 'coco_dev_ids.npy'), all_ann_ids)
    np.save(os.path.join(output_dir, 'coco_test_ids.npy'), all_ann_ids)
    np.save(os.path.join(output_dir, 'coco_restval_ids.npy'), np.array([]))
    
    print(f"Created dummy annotations in {output_dir}")

if __name__ == "__main__":
    create_dummy_annotations()
