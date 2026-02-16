### Quantum Configuration (if `employ_quantum_layer=True`)
```python
"quantum_NN": {
    "feature_map": zz_feature_map(feature_dimension=2),
    "ansatz": efficient_su2(num_qubits=2, reps=1),
    "execution_mode": "exact_simulator"  # "exact_simulator", "noisy_simulator", or "quantum_hardware"
}
```

### Execution Modes

- **exact_simulator**: Ideal quantum simulation (no noise, fastest)
- **noisy_simulator**: Simulates real quantum noise and errors
- **quantum_hardware**: Execute on actual IBM quantum computers (requires IBM Quantum account and access)

**Note**: Using real quantum hardware requires:
1. IBM Quantum account (free at quantum.ibm.com)
2. API token configuration in Qiskit
3. Access to available quantum processors
4. Significantly longer execution times due to queue wait times
