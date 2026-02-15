"""Hybrid Quantum-Classical Neural Network model"""

import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, Dropout2d, BatchNorm2d
from qiskit_machine_learning.connectors import TorchConnector


class QuantumHybridNet(Module):
    def __init__(self, config, qnn, n_classes):
        super().__init__()
        
        self.n_qubits = config["quantum_NN"]["feature_map"].num_qubits
        self.n_features = config["quantum_NN"]["feature_map"].num_parameters
        self.n_classes = n_classes
        self.employ_quantum = config["employ_quantum_layer"]
        
        self.conv1 = Conv2d(3, 64, kernel_size=5)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = Conv2d(64, 128, kernel_size=3)
        self.bn2 = BatchNorm2d(128)
        self.conv3 = Conv2d(128, 256, kernel_size=3)
        self.bn3 = BatchNorm2d(256)
        self.dropout = Dropout2d(p=0.3)
        self.fc1 = Linear(1024, 256)
        self.fc2 = Linear(256, 128)
        self.fc2_A = Linear(128, self.n_features)
        self.fc3_A = Linear(self.n_qubits, self.n_classes)
        self.fc2_B = Linear(128, 4)
        self.fc3_B = Linear(4, self.n_classes)
        self.qnn = TorchConnector(qnn)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        if self.employ_quantum:
            x = self.fc2_A(x)
            x = self.qnn(x)
            x = self.fc3_A(x)
        else:
            x = self.fc2_B(x)
            x = F.gelu(x)
            x = self.fc3_B(x)
        
        return x
