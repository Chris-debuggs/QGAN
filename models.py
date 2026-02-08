# models.py
"""
Neural Network Models for Hybrid Quantum-Classical GAN.
- PatchQuantumGenerator: Quantum generator using the patch method
- Discriminator: Classical MLP discriminator
"""
import torch
import torch.nn as nn

import config
import circuits


class PatchQuantumGenerator(nn.Module):
    """
    Quantum Generator using the Patch Method.
    
    Multiple sub-generators each produce a patch of the final image.
    Patches are concatenated to form the complete output.
    """
    
    def __init__(self, n_generators=config.N_GENERATORS, q_delta=1.0):
        """
        Args:
            n_generators: Number of sub-generators (patches)
            q_delta: Initial parameter spread for random initialization
        """
        super().__init__()
        
        # Each sub-generator has its own trainable parameters
        self.q_params = nn.ParameterList([
            nn.Parameter(
                q_delta * torch.rand(config.Q_DEPTH * config.N_QUBITS), 
                requires_grad=True
            )
            for _ in range(n_generators)
        ])
        
        self.n_generators = n_generators

    def forward(self, x):
        """
        Generate images from latent vectors.
        
        Args:
            x: Latent noise vectors [batch_size, N_QUBITS]
        
        Returns:
            Generated images [batch_size, TOTAL_PIXELS]
        
        NOTE: Sequential processing is required because quantum circuits
        don't batch like classical neural networks.
        """
        images = []
        
        # Process each sample in the batch
        for elem in x:
            patches = []
            
            # Each sub-generator produces one patch
            for params in self.q_params:
                patch = circuits.partial_measure(elem, params).float()
                patches.append(patch)
            
            # Concatenate patches into full image
            full_image = torch.cat(patches).to(config.DEVICE)
            images.append(full_image)
            
        return torch.stack(images)


class Discriminator(nn.Module):
    """
    Classical Discriminator Network (MLP).
    
    Takes flattened image and outputs probability of being real.
    """
    
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(config.TOTAL_PIXELS, 64),
            nn.ReLU(),
            
            # Hidden layer
            nn.Linear(64, 16),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(16, 1),
            nn.Sigmoid()  # Probability output [0, 1]
        )

    def forward(self, x):
        """
        Classify image as real or fake.
        
        Args:
            x: Flattened image [batch_size, TOTAL_PIXELS]
        
        Returns:
            Probability of being real [batch_size, 1]
        """
        return self.model(x)
