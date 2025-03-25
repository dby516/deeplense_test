"""
Lens Dataset Loader

This script defines a Lens Data class, which can:
 - Convert NPY images to torch tensors
 - Apply transformations (normalization)
 - Visualize a sample image

Dataset structure:
    Samples/
    ├── sample1.npy, sample2.npy, ...

Example usage: python data_proc.py

Bingyao Du, 3-17-2025
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class LensDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to dataset directory (e.g., "../Samples")
            transform: Transformations on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".npy")])

        if not self.data:
            raise RuntimeError(f"No NPY files found in {root_dir}")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Args:
            idx: Image index
        Returns:
            image: Image tensor of shape (1, H, W)
        """
        path = self.data[idx]
        image = np.load(path).astype(np.float32)

        if self.transform:
            image = self.transform(torch.tensor(image))

        return image

    def vis_item(self, idx):
        """ Visualize an image from the dataset """
        image = np.load(self.data[idx])

        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)  # Converts (1, H, W) → (H, W)

        plt.imshow(image, cmap="gray")
        plt.title(f"Sample {idx}")
        plt.axis("off")
        plt.show()

    def vis_img(image):
        """ Visualize an imag """
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)  # Converts (1, H, W) → (H, W)

        plt.imshow(image, cmap="gray")
        plt.title(f"Sample Image")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))  # Standard normalization
    ])
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Samples"))
    dataset = LensDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test loading
    for images in dataloader:
        print(f"Batch size: {images.shape}")
        break

    # Test visualization
    dataset.vis_item(0)
