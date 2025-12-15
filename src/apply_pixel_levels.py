import torch
import sys
from pathlib import Path
# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from get_mnist import get_mnist_raw
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import re

class Pixelate(nn.Module):
    def __init__(self, downscale_factor=4.0):

        super().__init__()
        self.downscale_factor = float(downscale_factor)

    def forward(self, img):

        C, H, W = img.shape
        
        h_small = max(1, int(round(H / self.downscale_factor)))
        w_small = max(1, int(round(W / self.downscale_factor)))

        img_small = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(h_small, w_small), mode="nearest"
        )
        img_big = torch.nn.functional.interpolate(
            img_small, size=(H, W), mode="nearest"
        )
        return img_big.squeeze(0)


def get_last_image_index(output_dir):
    """
    Find the highest index of existing image directories.
    Returns -1 if no images exist, otherwise returns the highest index found.
    """
    if not os.path.exists(output_dir):
        return -1
    
    max_index = -1
    pattern = re.compile(r'image_(\d+)_class_\d+')
    
    for item in os.listdir(output_dir):
        match = pattern.match(item)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    
    return max_index


def apply_pixelation_to_mnist(num_images=200, pixelate_levels=[1.5, 2.4, 3.2, 4.0, 4.9, 5.8, 7.0, 10], 
                              output_dir='pixelated_mnist', use_train=True, resume=True):

    # Load MNIST data
    print("Loading MNIST data...")
    train_images, train_labels, test_images, test_labels = get_mnist_raw()

    if use_train:
        images = train_images
        labels = train_labels
        dataset_name = "train"
    else:
        images = test_images
        labels = test_labels
        dataset_name = "test"
    
    # Check for existing images and determine starting index
    start_index = 0
    if resume:
        last_index = get_last_image_index(output_dir)
        if last_index >= 0:
            start_index = last_index + 1
            print(f"Found existing images up to index {last_index}. Resuming from index {start_index}...")
    
    # Select subset of images
    num_images = min(num_images, len(images))
    selected_indices = torch.randperm(len(images))[:num_images]
    selected_images = images[selected_indices]
    selected_labels = labels[selected_indices]
    
    print(f"Processing {num_images} images from {dataset_name} set (starting at index {start_index})...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    pixelate = Pixelate()
    
    for local_idx, (img, label) in enumerate(zip(selected_images, selected_labels)):
        global_idx = start_index + local_idx
        img_dir = os.path.join(output_dir, f"image_{global_idx:03d}_class_{label.item()}")
        os.makedirs(img_dir, exist_ok=True)
        
        img_with_channel = img.unsqueeze(0) 
        
        for pixelate_level in pixelate_levels:
            pixelate.downscale_factor = pixelate_level
            pixelated_img = pixelate(img_with_channel)

            img_np = pixelated_img.squeeze(0).detach().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            pil_img = Image.fromarray(img_np, mode='L')
            
            int_part = int(pixelate_level)
            dec_part = int((pixelate_level - int_part) * 10)
            filename = f"{int_part:02d}.{dec_part}_pixelate.png"
            filepath = os.path.join(img_dir, filename)
            pil_img.save(filepath)
        
        if (local_idx + 1) % 20 == 0:
            print(f"Processed {local_idx + 1}/{num_images} images (current index: {global_idx})...")
    
    print(f"Done! Generated {num_images} images starting from index {start_index}.")


if __name__ == "__main__":
    # Apply pixelation with specified levels
    # More granular at lower pixelation (where users can actually guess)
    # to capture more data points in the range where recognition happens
    pixelate_levels = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.2, 5.0, 6.5]
    apply_pixelation_to_mnist(
        num_images=1000,
        pixelate_levels=pixelate_levels,
        output_dir='pixelated_mnist',
        use_train=True,
        resume=True
    )