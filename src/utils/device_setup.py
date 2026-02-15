"""Device setup utilities for PyTorch and Qiskit execution"""

import torch
from qiskit_aer import AerSimulator
from qiskit_machine_learning.utils import algorithm_globals


def setup_devices(config):
    """
    Setup execution devices for classical and quantum parts.
    
    Args:
        config (dict): Configuration dictionary containing classical_device 
                       and quantum_NN execution_mode settings
        
    Returns:
        tuple: (torch_device, qiskit_device) - Configured devices for PyTorch and Qiskit
    """
    # Set seed for random generators
    algorithm_globals.random_seed = 42
    
    # Setup GPU Device for both classical part and quantum simulation part
    torch_device = torch.device("cpu")  # default
    qiskit_device = 'CPU'  # default
    
    if torch.cuda.is_available() and config["classical_device"] == "GPU":
        torch_device = torch.device("cuda")
        
        if 'GPU' in AerSimulator().available_devices():   # try to use GPU for quantum too
            qiskit_device = 'GPU'
        else:
            print("PyTorch on GPU but Qiskit GPU not available - using CPU for quantum")
    
    print(f"Execution device for the classical part: {torch_device}")
    if config["quantum_NN"]["execution_mode"] != "quantum_hardware":
        print(f"Execution device for simulation of quantum layer: {qiskit_device}")
    
    if torch_device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return torch_device, qiskit_device
