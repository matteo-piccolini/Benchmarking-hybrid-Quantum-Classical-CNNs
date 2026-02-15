"""Configuration validation utilities"""


def validate_config(config):
    """
    Validate configuration dictionary.

    Args:
        config (dict): Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    if config["quantum_NN"]["feature_map"].num_qubits != config["quantum_NN"]["ansatz"].num_qubits:
        raise ValueError(
            f"Qubit mismatch: feature_map has {config['quantum_NN']['feature_map'].num_qubits} qubits "
            f"but ansatz has {config['quantum_NN']['ansatz'].num_qubits} qubits"
        )

    valid_modes = ["exact_simulator", "noisy_simulator", "quantum_hardware"]
    if config["quantum_NN"]["execution_mode"] not in valid_modes:
        raise ValueError(
            f"Invalid execution_mode: {config['quantum_NN']['execution_mode']}. "
            f"Must be one of {valid_modes}"
        )

    valid_devices = ["CPU", "GPU"]
    if config["classical_device"] not in valid_devices:
        raise ValueError(
            f"Invalid classical_device: {config['classical_device']}. "
            f"Must be one of {valid_devices}"
        )

    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")

    if config["images_per_class"] <= 0:
        raise ValueError("images_per_class must be positive")

    if config["optimization"]["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")

    if config["optimization"]["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive")

    print("Configuration validated successfully")
