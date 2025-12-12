import torch
from get_mnist import get_mnist_raw
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image
import numpy as np

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


def apply_pixelation_to_mnist(num_images=200, pixelate_levels=[1.5, 2.4, 3.2, 4.0, 4.9, 5.8, 7.0, 10], 
                              output_dir='pixelated_mnist', use_train=True):

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
    
    # Select subset of images
    num_images = min(num_images, len(images))
    selected_indices = torch.randperm(len(images))[:num_images]
    selected_images = images[selected_indices]
    selected_labels = labels[selected_indices]
    
    print(f"Processing {num_images} images from {dataset_name} set...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    pixelate = Pixelate()
    
    for idx, (img, label) in enumerate(zip(selected_images, selected_labels)):
        img_dir = os.path.join(output_dir, f"image_{idx:03d}_class_{label.item()}")
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
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{num_images} images...")
    
    print("done")


if __name__ == "__main__":
    # Apply pixelation with specified levels
    pixelate_levels = [1.5, 2.4, 3.2, 4.0, 4.9, 5.8, 7.0, 10]
    apply_pixelation_to_mnist(
        num_images=200,
        pixelate_levels=pixelate_levels,
        output_dir='pixelated_mnist',
        use_train=True
    )

