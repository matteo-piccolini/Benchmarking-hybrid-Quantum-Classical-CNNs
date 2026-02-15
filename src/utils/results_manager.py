"""Results management utilities for saving and comparing experiments"""

import json
from datetime import datetime, timezone, timedelta


def save_results(config, metrics, training_time, results_file, actual_device=None, model_name=None):
    """
    Save experiment results to JSON file.
    If a run with the same configuration exists, replace it with the new one.
    
    Args:
        config: CONFIG dictionary
        metrics: dictionary with accuracy, precision, recall, f1, etc.
        training_time: total training time in seconds
        results_file: path to JSON file
    """
    ##################################################################################################
    # Create dictionary containing results merging the "config" dictionary, metrics, and training time provided in input
    new_result = {
        'timestamp': datetime.now(timezone(timedelta(hours=1))).strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'model_name': model_name if model_name else 'Unknown',
            'employ_quantum': config['employ_quantum_layer'],
            'classical_device': config['classical_device'],
            'actual_device': str(actual_device) if actual_device else config['classical_device'],
            'num_epochs': config['optimization']['num_epochs'],
            'learning_rate': config['optimization']['learning_rate'],
            'optimizer': config['optimization']['optimizer'],
            'batch_size': config['batch_size'],
            'images_per_class': config.get('images_per_class', 'N/A'),
            'num_qubits': config['quantum_NN']['feature_map'].num_qubits if config['employ_quantum_layer'] else 'N/A',
            'feature_map': config['quantum_NN']['feature_map'].name if config['employ_quantum_layer'] else 'N/A',
            'ansatz': config['quantum_NN']['ansatz'].name if config['employ_quantum_layer'] else 'N/A',
            'execution_mode': config['quantum_NN']['execution_mode'] if config['employ_quantum_layer'] else 'N/A'
        },
        'metrics': metrics,
        'training_time_seconds': training_time
    }
    
    ##################################################################################################
    # Load already existing results
    if results_file.exists():
        with open(results_file, 'r') as res_file:
            results = json.load(res_file)
    else:
        results = []
    
    # Check for duplicate configuration
    duplicate_index = None
    for i, result in enumerate(results):
        # Compare all config parameters
        if (result['configuration']['employ_quantum'] == new_result['configuration']['employ_quantum'] and
            result['configuration']['classical_device'] == new_result['configuration']['classical_device'] and
            result['configuration']['num_epochs'] == new_result['configuration']['num_epochs'] and
            result['configuration']['learning_rate'] == new_result['configuration']['learning_rate'] and
            result['configuration']['optimizer'] == new_result['configuration']['optimizer'] and
            result['configuration']['batch_size'] == new_result['configuration']['batch_size'] and
            result['configuration']['num_qubits'] == new_result['configuration']['num_qubits']):
            
            duplicate_index = i
            break
    
    # Replace duplicate or append new result
    if duplicate_index is not None:
        old_timestamp = results[duplicate_index]['timestamp']
        results[duplicate_index] = new_result
        print(f"Previous run from {old_timestamp} has been replaced")
    else:
        results.append(new_result)
        print()
        print(f"New configuration saved")
    
    # Save updated results (previous + current)
    with open(results_file, 'w') as res_file:
        json.dump(results, res_file, indent=2)
    
    print()
    print(f"Results saved to {results_file}")


def load_and_compare_results(results_file):
    """
    Load and display comparison of all previous runs
    
    Args:
        results_file: path to JSON file
    """
    ##################################################################################################
    # Check if previous results exist and load them if they do
    if not results_file.exists():
        print()
        print("No previous results found")
        return
    
    with open(results_file, 'r') as res_file:
        results = json.load(res_file)
    
    print()
    
    ##################################################################################################
    # Scan through results and print them for comparison
    print(f"{'='*80}")
    print(f"COMPARISON OF {len(results)} PREVIOUS RUN(S)")
    print(f"{'='*80}\n")
    
    for idx, result in enumerate(results, 1):
        quantum_str = "Quantum" if result['configuration']['employ_quantum'] else "Classical"
        device_str = result['configuration'].get('actual_device', result['configuration'].get('classical_device', 'N/A'))
        model_name = result['configuration'].get('model_name', 'N/A')
        
        print(f"Run {idx} - {result['timestamp']}")
        print(f"  Model: {model_name}")
        print(f"  Configuration: {quantum_str} | "
              f"Device: {device_str} | "
              f"Epochs: {result['configuration']['num_epochs']} | "
              f"LR: {result['configuration']['learning_rate']} | "
              f"Optimizer: {result['configuration']['optimizer']}")
        print(f"  Metrics:")
        print(f"    Accuracy: {result['metrics']['accuracy']*100:.2f}% | "
              f"Precision: {result['metrics']['precision']:.4f} | "
              f"Recall: {result['metrics']['recall']:.4f} | "
              f"F1: {result['metrics']['f1']:.4f}")
        print(f"  Training time: {result['training_time_seconds']:.2f}s")
        print()
