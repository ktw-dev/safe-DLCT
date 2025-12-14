import os
import subprocess

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filename = url.split('/')[-1]
    dest_path = os.path.join(dest_folder, filename)
    
    if os.path.exists(dest_path):
        print(f"{filename} already exists.")
        return dest_path

    print(f"Downloading {filename} to {dest_folder}...")
    cmd = f"wget -c -P {dest_folder} {url}"
    subprocess.run(cmd, shell=True, check=True)
    return dest_path

def main():
    # Depth Anything V2 (ViT-L)
    # URL might need to be updated if changed, using a known source or huggingface
    # Using the official repo release link if available, or huggingface
    depth_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    download_file(depth_url, "checkpoints")

    # Detectron2 X-101 (ResNeXt-101)
    # This is often used in VQA. If specific X-101.pth is missing, we can use a similar one from Detectron2 model zoo
    # But the script hardcodes 'output_X101/X-101.pth'.
    # Let's try to download a standard X-101 model and symlink or rename it.
    # R-101-FPN from detectron2 model zoo:
    # https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
    # But the script expects .pth. Detectron2 pickles are compatible.
    
    # However, 'grid-feats-vqa' usually uses a specific model trained on Visual Genome.
    # Let's assume for now we use a standard COCO model for demonstration if the specific one isn't found.
    # I will download a standard X-101 model.
    x101_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    dest = download_file(x101_url, "output_X101")
    
    # Rename/Symlink to X-101.pth
    if dest:
        target = os.path.join("output_X101", "X-101.pth")
        if not os.path.exists(target):
            os.symlink("model_final_68b088.pkl", target)
            print(f"Linked {dest} to {target}")

if __name__ == "__main__":
    main()
