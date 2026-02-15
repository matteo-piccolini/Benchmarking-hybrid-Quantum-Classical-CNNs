"""Data loading utilities for CIFAR-10 dataset"""

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_dataset(data_dir, samples_per_class, batch_size):
    """
    Load and filter CIFAR-10 dataset.
    
    Args:
        data_dir (Path): Directory to store/load dataset
        samples_per_class (int): Number of training samples per class
        batch_size (int): Batch size for DataLoader
        
    Returns:
        tuple: (X_train, X_test, train_loader, test_loader)
    """
    ##################################################################################################
    # Download train dataset
    # Data augmentation for training set
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # horizontal flip with 50% probability
        transforms.RandomCrop(32, padding=4),     # random crop with padding
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # random brightness/contrast
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10 mean/std
    ])
    
    X_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    original_num_samples_train = len(X_train)
    
    # Select only a user-defined number of train images per class
    filtered_train_idx = np.concatenate([np.where(np.array(X_train.targets) == i)[0][:samples_per_class] for i in range(len(X_train.classes))])
    X_train.data = X_train.data[filtered_train_idx]
    X_train.targets = np.array(X_train.targets)[filtered_train_idx]
    
    # Divide the reduced train dataset in batches
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    
    ##################################################################################################
    # Download test dataset
    # No augmentation for test set, only normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    X_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    original_num_samples_test = len(X_test)
    
    # Select only a user-defined number of test images per class
    filtered_test_idx = np.concatenate([np.where(np.array(X_test.targets) == i)[0][:samples_per_class//5] for i in range(len(X_test.classes))])
    X_test.data = X_test.data[filtered_test_idx]
    X_test.targets = np.array(X_test.targets)[filtered_test_idx]
    
    # Divide the reduced test dataset in batches
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)
    
    ##################################################################################################
    # Print dataset characteristics
    channels, height, width = X_train[0][0].shape
    print()
    print(f"Images type: {type(X_train[0][0])}")
    print(f"Number of channels per image: {channels}")
    print(f"Size of images: {height} x {width} pixels")
    
    print("-" * 80)
    print(f"Number of samples in the raw train dataset: {original_num_samples_train}")
    print(f"Number of samples in the raw test dataset: {original_num_samples_test}")
    print(f"Number of classes: {len(X_train.classes)}")
    print("Classes names: ", X_train.classes)
    
    print("-" * 80)
    print(f"Number of samples in the filtered train dataset: {len(X_train)}")
    print(f"Number of samples in the filtered test dataset: {len(X_test)}")
    print(f"Batch size: {batch_size}")
    
    return X_train, X_test, train_loader, test_loader
