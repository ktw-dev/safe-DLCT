import h5py
import numpy as np
import os

def generate_dummy_h5(filename='dummy_features.h5', num_images=5):
    with h5py.File(filename, 'w') as f:
        for i in range(num_images):
            image_id = 1000 + i
            
            # Create dummy features
            # features: (n_boxes, 2048)
            n_boxes = np.random.randint(5, 20)
            features = np.random.rand(n_boxes, 2048).astype(np.float32)
            f.create_dataset('%d_features' % image_id, data=features)
            
            # grids: (49, 2048)
            grids = np.random.rand(49, 2048).astype(np.float32)
            f.create_dataset('%d_grids' % image_id, data=grids)
            
            # boxes: (n_boxes, 4)
            boxes = np.random.rand(n_boxes, 4).astype(np.float32)
            f.create_dataset('%d_boxes' % image_id, data=boxes)
            
            # size: (1, 2)
            size = np.array([[640, 480]]).astype(np.int32)
            f.create_dataset('%d_size' % image_id, data=size)
            
            # mask: (n_boxes, 49)
            mask = np.random.randint(0, 2, (n_boxes, 49)).astype(np.float32)
            f.create_dataset('%d_mask' % image_id, data=mask)
            
            # cls_prob: (n_boxes, 81) - Optional but used in field.py
            cls_prob = np.random.rand(n_boxes, 81).astype(np.float32)
            f.create_dataset('%d_cls_prob' % image_id, data=cls_prob)

    print(f"Generated {filename} with {num_images} dummy images.")

if __name__ == "__main__":
    generate_dummy_h5()
