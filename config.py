# config.py
"""
Quantum GAN Configuration
All hyperparameters and device settings in one place.
"""
import torch
import pennylane as qml

# --- Reproducibility ---
SEED = 42

# --- Quantum Circuit Parameters ---
N_QUBITS = 5          # Total qubits per sub-generator
N_A_QUBITS = 1        # Ancillary qubits (traced out for non-linearity)
Q_DEPTH = 6           # Depth of the variational circuit
N_GENERATORS = 4      # Number of sub-generators (patches)

# --- Data Parameters ---
IMAGE_SIZE = 8        # Input image dimension (8x8)
BATCH_SIZE = 1        # Batch size (keep small for quantum simulation)
TARGET_DIGIT = 0      # Which digit to train on

# --- Training Parameters ---
LR_GENERATOR = 0.3    # Learning rate for generator
LR_DISCRIMINATOR = 0.01  # Learning rate for discriminator
NUM_EPOCHS = 50       # Number of training epochs
LOG_INTERVAL = 10     # Log every N epochs

# --- Derived Constants ---
PATCH_SIZE = 2 ** (N_QUBITS - N_A_QUBITS)  # 16 pixels per patch
TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE     # 64 total pixels

# --- Device Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Q_DEV = qml.device("lightning.qubit", wires=N_QUBITS)
