"""
Lens Dataset Loader

This script defines a Lens Data class, which can
 - load images and assign labels
 - convert NPY images to torch tensors
 - apply transformations(normalization)

Dataset structure:
    dataset/
    ├── train/
    │   ├── no/  # No Substructure
    │   │   ├── 1.npy
    │   │   ├── 2.npy
    │   │   └── ...
    │   ├── sphere/  # Subhalo Substructure
    │   ├── vort/  # Vortex Substructure
    ├── val/

Example usage: python data_proc.py

Bingyao Du, 3-8-2025
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

class LensDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: path to dataset directory(e.g., "/home/test1/dataset/train")
            transform: transformations on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.data = []

        # Load NPY files
        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            label = idx

            for file in os.listdir(class_path):
                if file.endswith(".npy"):
                    self.data.append((os.path.join(class_path, file), label))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Args:
            idx: image index
        Returns:
            image: image tensor of shape (1, H, W)
            label: class label(integer)
        """
        path, label = self.data[idx]
        image = np.load(path).astype(np.float32)
        # image = image.squeeze(0)
        # Convert NumPy to PIL Image
        # image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        image = torch.tensor(image)
        return image, label
        
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ]) # Standard normalization
    
    dataset = LensDataset("/root/autodl-fs/dataset/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test loading
    for images, labels in dataloader:
        print(f"Batch size: {images.shape}, Labels: {labels}")
        break


    # # Load the .npy image
    # image = np.load("dataset/train/sphere/5.npy")
    # if image.shape[0] == 1:  
    #     image = image.squeeze(0)  # Now shape: (150, 150)

    # # Display the image
    # plt.imshow(image, cmap="gray")  # Use 'gray' for grayscale images
    # plt.colorbar()
    # plt.title("Numpy Image")
    # plt.show()