"""Training utilities for the hybrid model"""

import time
import torch
from torch.nn import CrossEntropyLoss


def create_optimizer(model, config_dict, **kwargs):
    """
    Create optimizer from config dictionary.

    Args:
        model: PyTorch model
        config_dict: Configuration dictionary with optimization parameters
        **kwargs: Additional optimizer parameters

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    params = {"lr": config_dict["optimization"]["learning_rate"], **kwargs}
    if config_dict["optimization"]["optimizer"] == "LBFGS":
        params["max_iter"] = config_dict["optimization"]["num_epochs"]
    return getattr(torch.optim, config_dict["optimization"]["optimizer"])(model.parameters(), **params)


def single_epoch_training(model, train_loader, optimizer, loss_func, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer instance
        loss_func: Loss function
        device: Torch device (CPU/GPU)

    Returns:
        tuple: (epoch_loss, epoch_training_time)
    """
    model.train()
    losses = []
    start_epoch = time.time()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        if isinstance(optimizer, torch.optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                loss = loss_func(model(data), target)
                loss.backward()
                return loss
            losses.append(optimizer.step(closure).item())
        else:
            optimizer.zero_grad()
            loss = loss_func(model(data), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    epoch_time = time.time() - start_epoch
    return sum(losses)/len(losses), epoch_time


def train_model(model, train_loader, config_dict, device, loss_func=None):
    """
    Train model for multiple epochs.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        config_dict: Configuration dictionary
        device: Torch device (CPU/GPU)
        loss_func: Loss function (default: CrossEntropyLoss)

    Returns:
        tuple: (loss_list, total_training_time)
    """
    optimizer = create_optimizer(model, config_dict)
    epochs = 1 if isinstance(optimizer, torch.optim.LBFGS) else config_dict["optimization"]["num_epochs"]

    if not loss_func:
        loss_func = CrossEntropyLoss()

    losses = []
    start = time.time()

    for e in range(epochs):
        print(f"--- Epoch {e+1}/{epochs} ---")
        loss, epoch_time = single_epoch_training(model, train_loader, optimizer, loss_func, device)
        losses.append(loss)
        print(f"Epoch loss: {loss:.4f} | Epoch training time: {epoch_time:.2f}s\n")

    total_time = time.time() - start
    print(f"{'='*60}")
    print(f"Training Finished! Total Time: {total_time:.2f}s")
    print(f"{'='*60}\n")

    return losses, total_time
