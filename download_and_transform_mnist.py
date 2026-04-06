
import gzip
import struct
import urllib.request
from pathlib import Path
 
import numpy as np

 
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
 
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
 
DATA_DIR = Path("./mnist_data")
 
N_TRAIN = 1000
N_TEST  = 100
 
 
def download_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, fname in FILES.items():
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"  [skip]     {fname}")
        else:
            url = MNIST_BASE + fname
            print(f"  [download] {url}")
            urllib.request.urlretrieve(url, dest)
 
def read_images(gz_path: Path, count: int) -> np.ndarray:
    """Returns float32 array of shape (count, 784), values in [0, 1]."""
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(count * rows * cols)   # read only the first `count` images
    images = np.frombuffer(buf, dtype=np.uint8).reshape(count, rows * cols)
    return images.astype(np.float32) / 255.0
 
def read_labels(gz_path: Path, count: int) -> np.ndarray:
    """Returns float32 array of shape (count,), values 0-9."""
    with gzip.open(gz_path, "rb") as f:
        buf = f.read(count)                 # read only the first `count` labels
    return np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
 
def save_bin(arr: np.ndarray, path: Path):
    assert arr.dtype == np.float32
    arr.tofile(path)
    print(f"  [saved]    {path}  {arr.shape}  ({arr.nbytes / 1e3:.1f} KB)")
 
if __name__ == "__main__":
    print("Downloading MNIST...")
    download_all()
 
    print(f"\nParsing and saving binary files (train={N_TRAIN}, test={N_TEST})...")
 
    train_images = read_images(DATA_DIR / FILES["train_images"], N_TRAIN)
    train_labels = read_labels(DATA_DIR / FILES["train_labels"], N_TRAIN)
    test_images  = read_images(DATA_DIR / FILES["test_images"],  N_TEST)
    test_labels  = read_labels(DATA_DIR / FILES["test_labels"],  N_TEST)
 
    save_bin(train_images, DATA_DIR / "train_images.bin")
    save_bin(train_labels, DATA_DIR / "train_labels.bin")
    save_bin(test_images,  DATA_DIR / "test_images.bin")
    save_bin(test_labels,  DATA_DIR / "test_labels.bin")
 