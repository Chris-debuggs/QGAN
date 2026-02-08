# utils.py
"""
Utility functions for visualization and checkpointing.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import config


def set_seed(seed=config.SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def visualize_training_progress(images, step_interval=config.LOG_INTERVAL, save_path=None):
    """
    Display generated images at each checkpoint.
    
    Args:
        images: List of generated images (numpy arrays)
        step_interval: Epochs between each saved image
        save_path: Optional path to save the figure
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(2 * n_images, 2))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Epoch {i * step_interval}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training progress to {save_path}")
    
    plt.show()


def save_checkpoint(generator, discriminator, epoch, path='checkpoint.pt'):
    """
    Save model checkpoint.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        epoch: Current epoch number
        path: Save path
    """
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(generator, discriminator, path='checkpoint.pt'):
    """
    Load model checkpoint.
    
    Args:
        generator: Generator model (will be modified in-place)
        discriminator: Discriminator model (will be modified in-place)
        path: Checkpoint path
    
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(path, map_location=config.DEVICE)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']


def generate_samples(generator, n_samples=8):
    """
    Generate sample images from the generator.
    
    Args:
        generator: Trained generator model
        n_samples: Number of samples to generate
    
    Returns:
        numpy array of generated images [n_samples, IMAGE_SIZE, IMAGE_SIZE]
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.rand(n_samples, config.N_QUBITS, device=config.DEVICE) * np.pi / 2
        images = generator(noise)
        images = images.view(n_samples, config.IMAGE_SIZE, config.IMAGE_SIZE)
    generator.train()
    return images.cpu().numpy()
