# Benchmarking Hybrid Quantum-Classical CNNs

A flexible benchmarking framework for comparing classical and quantum-hybrid neural network architectures on image classification tasks.

## 🎯 Project Goal

This project provides a **modular benchmarking infrastructure** to systematically compare different neural network architectures, optimizers, and hyperparameters. The framework is designed for flexibility - the included hybrid CNN-QNN architecture is just an example, and users can easily plug in their own models.

**Key Philosophy:**
- **Benchmarking > Specific Results**: The framework matters more than any single architecture
- **Flexibility First**: Easy to modify, extend, and experiment with
- **Reproducibility**: Structured configuration and automatic results tracking
- **Fair Comparison**: Standardized evaluation metrics and visualization

## ✨ Key Features

### 🔧 Flexibility & Modularity
- **Pluggable Architectures**: Easily add your own model classes - the provided hybrid CNN is just an example
- **Configurable Quantum Layer**: Swap feature maps, ansätze, and execution modes (simulators or real quantum hardware)
- **Any PyTorch Optimizer**: Supports all torch.optim optimizers (Adam, SGD, RMSprop, AdamW, LBFGS, etc.)
- **Simple Configuration**: Just modify the CONFIG dictionary - everything else is automatic

### 📊 Comprehensive Benchmarking
- **Automatic Results Tracking**: JSON-based storage with timestamps and full configuration
- **Pareto Front Analysis**: Identify optimal accuracy/time trade-offs (min 60% accuracy threshold)
- **Visual Comparison**: Interactive plots comparing all runs
- **Detailed Metrics**: Accuracy, precision, recall, F1-score for every experiment

### ⚡ Performance
- **GPU Acceleration**: CUDA support for faster training
- **Efficient Data Loading**: PyTorch DataLoader with configurable batch sizes
- **Progress Monitoring**: Real-time loss and timing information

## 📁 Project Structure
```
Benchmarking_hybrid_Quantum-Classical_CNNs/
├── src/
│   ├── models/
│   │   └── hybrid_model.py          # Example hybrid CNN-QNN (easily replaceable)
│   ├── training/
│   │   ├── trainer.py               # Training loop with multi-optimizer support
│   │   └── evaluation.py            # Evaluation and metrics calculation
│   ├── utils/
│   │   ├── data_utils.py            # Data loading (currently CIFAR-10)
│   │   ├── device_setup.py          # GPU/CPU device configuration
│   │   ├── quantum_utils.py         # QNN setup and Qiskit integration
│   │   ├── results_manager.py       # Results storage and retrieval
│   │   └── visualization.py         # Plotting and Pareto front analysis
├── results/
│   └── benchmark_results.json       # All experiment results
├── datasets/                         # Dataset cache (auto-downloaded)
├── checkpoints/                      # Model checkpoints (optional)
├── Benchmarking_hybrid_Quantum_Classical_CNNs.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/Benchmarking_hybrid_Quantum-Classical_CNNs.git
cd Benchmarking_hybrid_Quantum-Classical_CNNs
pip install -r requirements.txt
```

### Run an Experiment

**It's that simple**: Just modify the CONFIG dictionary and run the cells in the notebook. Everything else (data loading, training, evaluation, results saving) is handled automatically.
```python
# 1. Configure your experiment
CONFIG = {
    "classical_device": "GPU",
    "batch_size": 30,
    "images_per_class": 500,
    "employ_quantum_layer": False,
    "optimization": {
        "optimizer": "Adam",
        "learning_rate": 0.002,
        "num_epochs": 100
    },
    "quantum_NN": {
        "feature_map": zz_feature_map(feature_dimension=2),
        "ansatz": efficient_su2(num_qubits=2, reps=1),
        "execution_mode": "exact_simulator"
    }
}

# 2. Run the notebook cells - that's it!
# Data loading, training, evaluation, and results saving are automatic
```

Want to try a different optimizer? Just change `"optimizer": "RMSprop"` and adjust the learning rate. Want quantum? Set `"employ_quantum_layer": True`. The framework handles the rest.

### Visualize Results
```python
from src.utils.visualization import plot_results_comparison

plot_results_comparison(RESULTS_FILE)
```

Generates:
- **Scatter plot**: F1-score vs Training Time with Pareto front (accuracy ≥ 60%)
- **Comparison table**: Top 10 classical + Top 10 quantum + Pareto runs (highlighted in orange)

## 🔧 Customization Guide

### Adding Your Own Architecture

The included `QuantumHybridNet` is just an example. To benchmark your own model:

1. **Create your model class** in `src/models/`:
```python
class YourCustomNet(nn.Module):
    def __init__(self, config, qnn=None, n_classes=10):
        super().__init__()
        # Your architecture here
        
    def forward(self, x):
        # Your forward pass
        return x
```

2. **Use it in experiments**:
```python
from src.models.your_model import YourCustomNet

model = YourCustomNet(config=CONFIG, qnn=qnn, n_classes=10)
# Training and evaluation work the same way
```

3. **Results automatically tracked** with model name for comparison

### Modifying the Example Hybrid Architecture

The example CNN-QNN in `src/models/hybrid_model.py` has:
- **Classical path**: Easily modify convolutional layers, channels, dropout rates
- **Quantum path**: Swap QNN by changing `feature_map`, `ansatz`, `execution_mode` in CONFIG
```python
# Example: Different quantum circuit
CONFIG["quantum_NN"] = {
    "feature_map": ry_feature_map(feature_dimension=4),
    "ansatz": real_amplitudes(num_qubits=4, reps=2),
    "execution_mode": "noisy_simulator"
}
```

### Optimizer Configurations

The framework supports **any PyTorch optimizer** from `torch.optim`. Just specify the name in CONFIG.

#### First-Order Optimizers (Adam, SGD, RMSprop, AdamW, etc.)
```python
"optimization": {
    "optimizer": "Adam",       # Or "SGD", "RMSprop", "AdamW", "Adagrad", etc.
    "learning_rate": 0.001,    # Typical range: 0.0001-0.01
    "num_epochs": 100
}
```

Common learning rates by optimizer:
- **Adam**: 0.001-0.002
- **SGD**: 0.01-0.1
- **RMSprop**: 0.001-0.01
- **AdamW**: 0.001-0.003

#### Second-Order Optimizers (LBFGS)
```python
"optimization": {
    "optimizer": "LBFGS",
    "learning_rate": 1.0,      # Higher LR: 0.1-1.0
    "num_epochs": 50           # Becomes max_iter (auto-set to 1 epoch)
}
# Note: Use larger batch_size (≈ half dataset) for LBFGS
CONFIG["batch_size"] = 2500
```

**LBFGS specifics:**
- Automatically sets training to 1 epoch with `num_epochs` as `max_iter`
- Requires larger batch sizes for convergence
- More memory intensive but can find better minima

### Quantum Execution Modes

When `employ_quantum_layer=True`, choose your execution mode:
```python
"quantum_NN": {
    "feature_map": zz_feature_map(feature_dimension=2),
    "ansatz": efficient_su2(num_qubits=2, reps=1),
    "execution_mode": "exact_simulator"  # Choose mode here
}
```

#### Available Modes:

**1. exact_simulator** (Default)
- Ideal quantum simulation with no noise
- Fastest execution
- Perfect for algorithm development and testing

**2. noisy_simulator**
- Simulates realistic quantum noise and decoherence
- Models actual quantum hardware behavior
- Good for testing noise resilience

**3. quantum_hardware**
- Executes on real IBM quantum computers
- Requires IBM Quantum account (free at [quantum.ibm.com](https://quantum.ibm.com))
- Save your API token with Qiskit to enable access
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```

## 📊 Results Management

### Automatic Saving
```python
from src.utils.results_manager import save_results

save_results(CONFIG, metrics, training_time, RESULTS_FILE, device, model_name)
```

Results stored in JSON with:
- Full configuration (optimizer, LR, batch size, quantum settings, etc.)
- All metrics (accuracy, precision, recall, F1)
- Training time
- Timestamp
- Model name for multi-architecture comparison

### Comparing Results
```python
from src.utils.results_manager import load_and_compare_results

load_and_compare_results(RESULTS_FILE)
```

Prints formatted comparison of all previous runs.

## 🎨 Visualization Features

### Pareto Front Calculation
- Identifies optimal accuracy/time trade-offs
- **Minimum accuracy threshold: 60%** (prevents random guessing from dominating)
- Highlighted in orange in comparison table

### Legend
- Blue dots: Quantum-hybrid models
- Red dots: Classical models  
- Dashed line: Pareto front

## 🔬 Adapting to Other Datasets

The framework currently uses CIFAR-10 but can be adapted to other image datasets:

### What Needs Modification:
1. **Data loading** (`src/utils/data_utils.py`):
   - Change dataset class (e.g., `torchvision.datasets.MNIST`)
   - Update normalization values for new dataset

2. **Model architecture** (`src/models/hybrid_model.py`):
   - Adjust `in_channels` for grayscale (1) vs RGB (3)
   - Modify `n_classes` for different number of categories
   - Consider input dimensions (28x28 vs 32x32 affects pooling)

3. **Configuration**:
   - Add dataset-specific parameters to CONFIG if needed

### Example: MNIST Adaptation
```python
# In data_utils.py: Replace CIFAR10 with MNIST
# In hybrid_model.py: Change Conv2d(3, 64) to Conv2d(1, 64)
# Update CONFIG for 28x28 images if needed
```

The modular structure makes these changes straightforward.

## ⚙️ Configuration Reference

### Essential Parameters
```python
CONFIG = {
    "classical_device": "GPU",           # "GPU" or "CPU"
    "batch_size": 30,                    # Batch size for training
    "images_per_class": 500,             # Samples per class
    "employ_quantum_layer": False,       # Enable quantum path
    "optimization": {
        "optimizer": "Adam",              # Any torch.optim optimizer
        "learning_rate": 0.002,
        "num_epochs": 100
    }
}
```

### Quantum Configuration (if `employ_quantum_layer=True`)
```python
"quantum_NN": {
    "feature_map": zz_feature_map(feature_dimension=2),
    "ansatz": efficient_su2(num_qubits=2, reps=1),
    "execution_mode": "exact_simulator"  # Or "noisy_simulator", "quantum_hardware"
}
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` (30 → 20)
- Reduce `images_per_class` (500 → 250)

### Slow Training
- Verify GPU: `!nvidia-smi` (Colab) or `torch.cuda.is_available()`
- Check device assignment in results

### LBFGS Not Converging
- Increase `batch_size` to ~50% of dataset
- Reduce `learning_rate` (1.0 → 0.5 or 0.1)

### Quantum Hardware Connection Issues
- Verify your IBM Quantum account and token are correctly configured
- Check connection at [quantum.ibm.com](https://quantum.ibm.com)

## 🤝 Contributing

Contributions welcome! Ideas:
- New model architectures (different CNNs, transformers, etc.)
- Additional quantum circuit designs
- Support for more datasets
- Enhanced visualization options
- Distributed training support

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Qiskit Machine Learning**: Quantum computing framework
- **PyTorch**: Deep learning infrastructure
- **CIFAR-10**: Dataset by Alex Krizhevsky

---

**Note**: This is a benchmarking framework. The included hybrid CNN-QNN architecture is provided as an example and starting point - feel free to replace it with your own models for comparison!
