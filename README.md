# Quantum GAN - Patch Method

A hybrid quantum-classical Generative Adversarial Network using PennyLane and PyTorch.

## Overview

This implementation uses the **Patch Method** to overcome qubit limitations:
- Multiple quantum sub-generators each produce a small "patch"
- Patches are concatenated to form the complete image
- A classical discriminator learns to distinguish real from generated images

## Project Structure

```
quantum_gan_mnist/
├── config.py      # Hyperparameters and device settings
├── dataset.py     # Data loading (sklearn digits)
├── circuits.py    # Quantum circuits (PennyLane)
├── models.py      # Generator & Discriminator
├── train.py       # Training loop
├── utils.py       # Visualization & checkpointing
└── README.md      # This file
```

## Quick Start

```bash
# Install dependencies
pip install torch pennylane pennylane-lightning scikit-learn matplotlib

# Run training
python train.py
```

## Configuration

Edit `config.py` to modify:
- `N_QUBITS`: Total qubits per sub-generator (default: 5)
- `N_GENERATORS`: Number of patches (default: 4)
- `Q_DEPTH`: Circuit depth (default: 6)
- `NUM_EPOCHS`: Training epochs (default: 50)

## Architecture

### Quantum Generator
```
Noise → [RY Encoding] → [Variational Layers (RY + CZ)] → [Partial Measure] → Patch
                                                                              ↓
                              [4 Sub-generators] ──────────────────────→ Concatenate → 64px Image
```

### Classical Discriminator
```
64px Image → Linear(64→64) → ReLU → Linear(64→16) → ReLU → Linear(16→1) → Sigmoid → P(real)
```

## Adapting for Drug Discovery

To use this for molecular generation:

1. **Replace dataset**: Load molecular fingerprints instead of digits
2. **Adjust dimensions**: Set `N_GENERATORS` based on fingerprint size
3. **Modify discriminator**: Add property prediction heads

Example config changes:
```python
FINGERPRINT_SIZE = 1024  # Morgan fingerprint bits
N_GENERATORS = 64        # 1024 / 16 = 64 patches
```

## References

- [PennyLane Quantum GAN Tutorial](https://pennylane.ai/qml/demos/tutorial_quantum_gans/)
- Huang et al. "Experimental Quantum Generative Adversarial Networks" (2020) 
