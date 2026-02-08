# circuits.py
"""
Quantum Circuit Definitions for the Patch Method Generator.
Uses PennyLane for quantum simulation with parameter-shift gradients.
"""
import pennylane as qml
import torch

import config


@qml.qnode(config.Q_DEV, diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    """
    Variational quantum circuit for image generation.
    
    Args:
        noise: Latent vector encoded as rotation angles [N_QUBITS]
        weights: Trainable parameters [Q_DEPTH * N_QUBITS]
    
    Returns:
        Probability distribution over all 2^N_QUBITS basis states
    """
    weights = weights.reshape(config.Q_DEPTH, config.N_QUBITS)
    
    # Encoder: Embed latent vector as RY rotations
    for i in range(config.N_QUBITS):
        qml.RY(noise[i], wires=i)
        
    # Variational Layers
    for layer in range(config.Q_DEPTH):
        # Parameterized rotations
        for qubit in range(config.N_QUBITS):
            qml.RY(weights[layer][qubit], wires=qubit)
        
        # Entanglement: Linear chain of CZ gates
        for qubit in range(config.N_QUBITS - 1):
            qml.CZ(wires=[qubit, qubit + 1])
            
    # Return probabilities of all basis states
    return qml.probs(wires=list(range(config.N_QUBITS)))


def partial_measure(noise, weights):
    """
    Implements the Patch Method's non-linearity via partial measurement.
    
    The ancillary qubit is traced out, and normalization introduces
    the non-linear behavior needed for GAN training.
    
    Args:
        noise: Latent vector [N_QUBITS]
        weights: Circuit parameters [Q_DEPTH * N_QUBITS]
    
    Returns:
        Normalized probabilities representing PATCH_SIZE pixel values
    """
    # Get raw probabilities from quantum circuit
    probs = quantum_circuit(noise, weights)
    
    # 1. Trace out ancilla: Keep only first half of probabilities
    #    (corresponds to measuring ancilla in |0⟩ state)
    n_output = 2 ** (config.N_QUBITS - config.N_A_QUBITS)
    probs_given_0 = probs[:n_output]
    
    # 2. Normalize: This division creates non-linearity
    probs_given_0 = probs_given_0 / torch.sum(probs_given_0)
    
    # 3. Post-process: Rescale so max value is 1.0 (like pixel intensity)
    return probs_given_0 / torch.max(probs_given_0)
