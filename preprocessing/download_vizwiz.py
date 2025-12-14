import os
import subprocess

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filename = url.split('/')[-1]
    dest_path = os.path.join(dest_folder, filename)
    
    print(f"Downloading {filename} to {dest_folder}...")
    # Use wget for robust downloading
    cmd = f"wget -c -P {dest_folder} {url}"
    subprocess.run(cmd, shell=True, check=True)
    return dest_path

def unzip_file(file_path, dest_folder):
    print(f"Unzipping {file_path}...")
    cmd = f"unzip -o {file_path} -d {dest_folder}"
    subprocess.run(cmd, shell=True, check=True)

def main():
    base_dir = "data/vizwiz"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    urls = [
        "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip",
        "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
        "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
        "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
    ]

    for url in urls:
        try:
            file_path = download_file(url, base_dir)
            unzip_file(file_path, base_dir)
            # Optional: Remove zip file to save space
            # os.remove(file_path) 
        except subprocess.CalledProcessError as e:
            print(f"Error downloading or unzipping {url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
