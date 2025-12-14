import h5py
import sys

def check_h5(path):
    try:
        f = h5py.File(path, 'r')
        print(f"Opened {path} successfully.")
        keys = list(f.keys())
        print(f"Total keys: {len(keys)}")
        if len(keys) > 0:
            print(f"Sample key: {keys[0]}")
            print(f"Shape: {f[keys[0]].shape}")
        f.close()
    except Exception as e:
        print(f"Error opening {path}: {e}")

if __name__ == "__main__":
    check_h5(sys.argv[1])
