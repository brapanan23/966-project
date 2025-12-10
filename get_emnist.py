# Simple get EMNIST dataset script using TensorFlow Datasets
import tensorflow_datasets as tfds
import numpy as np
import torch
import gzip
import struct
import os

def load_idx_file(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        
        if magic == 2051:  
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

def get_emnist_from_raw_data(split='letters', data_dir='./raw_data'):
    base_path = os.path.join(data_dir, 'downloads', 'extracted', 
                            'ZIP.biometrics.nist.gov_cs_links_EMNIST_gzip-5u2fjN3KpzAuJXk7PNtLPNb6LcJaTw1ZM6ioBn82o4.zip',
                            'gzip')
    
    train_images_path = os.path.join(base_path, f'emnist-{split}-train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(base_path, f'emnist-{split}-train-labels-idx1-ubyte.gz')
    
    test_images_path = os.path.join(base_path, f'emnist-{split}-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(base_path, f'emnist-{split}-test-labels-idx1-ubyte.gz')
    
    print(f"Loading from raw_data directory...")
    print(f"  Train images: {train_images_path}")
    print(f"  Train labels: {train_labels_path}")
    print(f"  Test images: {test_images_path}")
    print(f"  Test labels: {test_labels_path}")
    
    train_images = load_idx_file(train_images_path)
    train_labels = load_idx_file(train_labels_path)
    test_images = load_idx_file(test_images_path)
    test_labels = load_idx_file(test_labels_path)
    
    print(f"Loaded: Train {train_images.shape}, Test {test_images.shape}")
    
    return train_images, train_labels, test_images, test_labels

def fix_emnist_orientation(imgs):

    imgs = np.transpose(imgs, (0, 2, 1))  # (N, H, W) -> (N, W, H)
    return imgs.copy()

def get_emnist_torch_from_raw_data(split='letters', data_dir='./raw_data', fix_orientation=True):
    train_images, train_labels, test_images, test_labels = get_emnist_from_raw_data(split, data_dir)
    
    if fix_orientation:
        train_images = fix_emnist_orientation(train_images)
        test_images = fix_emnist_orientation(test_images)
    
    train_images = torch.from_numpy(train_images).float() / 255.0
    train_labels = torch.from_numpy(train_labels).long()
    test_images = torch.from_numpy(test_images).float() / 255.0
    test_labels = torch.from_numpy(test_labels).long()
    
    print(f"Train: {train_images.shape}, {train_labels.shape}")
    print(f"Test: {test_images.shape}, {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels

def get_emnist_raw():

    ds = tfds.load('emnist/letters', data_dir='./data', download=True)
    return ds['train'], ds['test']

def get_emnist_torch():
    ds = tfds.load('emnist/letters', data_dir='./data', download=True)
    
    train = ds['train']
    test = ds['test']
    
    train_images = np.array([ex['image'].numpy() for ex in train])
    train_labels = np.array([ex['label'].numpy() for ex in train])
    test_images = np.array([ex['image'].numpy() for ex in test])
    test_labels = np.array([ex['label'].numpy() for ex in test])
    
    if len(train_images.shape) == 3:  # (N, H, W)
        train_images = fix_emnist_orientation(train_images)
        test_images = fix_emnist_orientation(test_images)
    elif len(train_images.shape) == 4:  # (N, H, W, C)
        train_images = np.transpose(train_images, (0, 2, 1, 3))
        train_images = np.flip(train_images, axis=1).copy()  # Flip along width
        test_images = np.transpose(test_images, (0, 2, 1, 3))
        test_images = np.flip(test_images, axis=1).copy()
    
    train_images = torch.from_numpy(train_images).float() / 255.0
    train_labels = torch.from_numpy(train_labels).long()
    test_images = torch.from_numpy(test_images).float() / 255.0
    test_labels = torch.from_numpy(test_labels).long()
    
    print(f"Train: {train_images.shape}, {train_labels.shape}")
    print(f"Test: {test_images.shape}, {test_labels.shape}")
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_emnist_torch()
