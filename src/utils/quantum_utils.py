"""Quantum circuit and QNN setup utilities"""

from qiskit.circuit.library import zz_feature_map, efficient_su2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.circuit.library import qnn_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


def setup_quantum_circuit(feature_map, ansatz):
    """
    Setup quantum circuit with feature map and ansatz.
    
    Args:
        feature_map: Qiskit feature map circuit
        ansatz: Qiskit ansatz circuit
        
    Returns:
        tuple: (qc, feature_map_params, ansatz_params, observables)
    """
    ##################################################################################################
    # Set up a circuit
    qc, feature_map_params, ansatz_params = qnn_circuit(feature_map=feature_map, ansatz=ansatz)
    
    print()
    print(f"Number of input parameters: {len(feature_map_params)}")
    print(f"Number of weight parameters: {len(ansatz_params)}\n")
    
    ##################################################################################################
    # Define a list of observables to be measured. Here we measure Z on each qubit, obtaining num_qubits values as output
    n_qubits = feature_map.num_qubits
    observables = [
        SparsePauliOp.from_list([("I" * i + "Z" + "I" * (n_qubits - 1 - i), 1)])
        for i in range(n_qubits)
    ]
    
    print(f"Measured observables: {observables}")
    
    return qc, feature_map_params, ansatz_params, observables


def setup_estimator(execution_mode, qiskit_device):
    """
    Setup estimator primitive based on execution mode.
    
    Args:
        execution_mode (str): One of "exact_simulator", "noisy_simulator", "quantum_hardware"
        qiskit_device (str): Device for simulation ("CPU" or "GPU")
        
    Returns:
        tuple: (estimator, pass_manager)
    """
    ##################################################################################################
    # Choose backends, transpile circuit, and define Estimator primitives according to the desired execution mode
    
    # Exact simulator (no noise): primitive - Aer EstimatorV2 (automatically relies on AerSimulator backend)
    if execution_mode == "exact_simulator":
        estimator = AerEstimator()
        estimator.options.backend_options = {
            'method': 'statevector',   # method suitable for exact simulations
            'device': qiskit_device
        }
        
        pass_manager = None
        
        print(f"Execution mode: {execution_mode}. Quantum Backend Configuration:")
        print(f"  Primitive: Aer EstimatorV2")
        print(f"  Internal backend: AerSimulator")
        print(f"  Method: {estimator.options.backend_options['method']}")
        print(f"  Device: {estimator.options.backend_options['device']}")
    
    # Noisy simulator: primitive - Aer EstimatorV2 (automatically relies on AerSimulator backend)
    elif execution_mode == "noisy_simulator":
        noisy_backend = FakeManilaV2()   # backend providing the noise and transpilation model for realistic simulation
        
        estimator = AerEstimator()
        estimator.options.backend_options = {
            'method': 'density_matrix',   # method suitable for noisy simulations
            'noise_model': NoiseModel.from_backend(noisy_backend),
            'device': qiskit_device
        }
        
        pass_manager = generate_preset_pass_manager(backend = noisy_backend, optimization_level = 3)
        
        print(f"Execution mode: {execution_mode}. Quantum Backend Configuration:")
        print(f"  Primitive: Aer EstimatorV2")
        print(f"  Internal backend: AerSimulator")
        print(f"  Fake backend: {noisy_backend.name}")
        print(f"  Noise model: from {noisy_backend.name}")
        print(f"  Method: {estimator.options.backend_options.get('method', 'default')}")
        print(f"  Device: {estimator.options.backend_options.get('device', 'CPU')}")
    
    # Real quantum hardware usage: primitive - runtime EstimatorV2, backend - least_busy
    elif execution_mode == "quantum_hardware":
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False, operational=True)
        
        pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
        estimator = Estimator(mode=backend)
        
        # Set error suppression/mitigation techniques and maximum execution time
        estimator.options.resilience_level = 3   # for error mitigation
        estimator.options.dynamical_decoupling.enable = True   # for error suppression
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.max_execution_time = 180   # set 3 minutes of maximum execution time
        
        print("Quantum Backend Configuration:")
        print(f"  Primitive: Runtime EstimatorV2")
        print(f"  Backend: {backend.name}")
        print(f"  Error mitigation: Level {estimator.options.resilience_level}")
        print(f"  Dynamical decoupling: {estimator.options.dynamical_decoupling.sequence_type}")
        print(f"  Max execution time: {estimator.options.max_execution_time}s")
        print(f"  Default shots: {estimator.options.default_shots}")
    
    return estimator, pass_manager


def create_qnn(qc, feature_map_params, ansatz_params, estimator, pass_manager, observables):
    """
    Create EstimatorQNN.
    
    Args:
        qc: Quantum circuit
        feature_map_params: Feature map parameters
        ansatz_params: Ansatz parameters
        estimator: Qiskit estimator primitive
        pass_manager: Transpiler pass manager
        observables: List of observables to measure
        
    Returns:
        EstimatorQNN: Configured quantum neural network
    """
    ##################################################################################################
    # Create EstimatorQNN
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        estimator=estimator,
        input_params=feature_map_params,
        weight_params=ansatz_params,
        input_gradients=True   # needed for hybrid backpropagation
    )
    
    return qnn
