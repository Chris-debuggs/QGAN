# dataset.py
"""
Dataset handling for Quantum GAN.
Uses scikit-learn's built-in digits dataset (8x8 images).
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits

import config


class DigitsDataset(Dataset):
    """
    Loads 8x8 digit images from Scikit-Learn.
    Filters for a specific digit to simplify training.
    """
    
    def __init__(self, digit=config.TARGET_DIGIT):
        """
        Args:
            digit: Which digit (0-9) to filter for training.
        """
        digits = load_digits()
        
        # Filter for the target digit
        mask = digits.target == digit
        self.data = digits.data[mask]
        
        # Normalize pixel values to [0, 1]
        self.data = self.data / 16.0
        
        print(f"Loaded {len(self.data)} samples of digit '{digit}'")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return flattened image as tensor
        img = torch.tensor(self.data[idx], dtype=torch.float32)
        return img, 0  # Label unused in GAN training


def get_dataloader(batch_size=config.BATCH_SIZE, shuffle=True):
    """Create a DataLoader for the digits dataset."""
    dataset = DigitsDataset()
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True  # Ensure consistent batch sizes
    )
