import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import numpy as np
import struct
import os
from torchvision import datasets, transforms

def get_mnist_torch(data_dir='./mnist_data'):
    """Load MNIST using torchvision datasets and convert to PyTorch tensors"""
    transform = transforms.ToTensor()
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_images = torch.stack([img for img, _ in train_dataset])
    train_labels = torch.tensor([label for _, label in train_dataset])
    test_images = torch.stack([img for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])
    
    if len(train_images.shape) == 4 and train_images.shape[1] == 1:
        train_images = train_images.squeeze(1)
        test_images = test_images.squeeze(1)
    
    print(f"Train: {train_images.shape}, {train_labels.shape}")
    print(f"Test: {test_images.shape}, {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels

def load_idx_file(filepath):
    """Load IDX file (MNIST format) - files are not gzipped"""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        
        if magic == 2051:  # Images
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
            return images
        elif magic == 2049:  # Labels
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        else:
            raise ValueError(f"Unknown magic number: {magic}")

def get_mnist_raw(data_dir='./mnist_data'):
    """Load MNIST from raw IDX files in mnist_data directory"""
    base_path = os.path.join(data_dir, 'MNIST', 'raw')
    
    train_images_path = os.path.join(base_path, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(base_path, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(base_path, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(base_path, 't10k-labels-idx1-ubyte')
    
    print(f"Loading from mnist_data directory...")
    print(f"  Train images: {train_images_path}")
    print(f"  Train labels: {train_labels_path}")
    print(f"  Test images: {test_images_path}")
    print(f"  Test labels: {test_labels_path}")
    
    train_images = load_idx_file(train_images_path)
    train_labels = load_idx_file(train_labels_path)
    test_images = load_idx_file(test_images_path)
    test_labels = load_idx_file(test_labels_path)
    
    # Convert to torch tensors
    train_images = torch.from_numpy(train_images.copy()).float() / 255.0
    train_labels = torch.from_numpy(train_labels.copy()).long()
    test_images = torch.from_numpy(test_images.copy()).float() / 255.0
    test_labels = torch.from_numpy(test_labels.copy()).long()
    
    print(f"Loaded: Train {train_images.shape}, Test {test_images.shape}")
    print(f"Train: {train_images.shape}, {train_labels.shape}")
    print(f"Test: {test_images.shape}, {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_mnist_torch()

