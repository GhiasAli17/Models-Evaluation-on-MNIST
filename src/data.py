import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split


def load_data_nn(batch_size):
    """
    Loads and prepares data for the Neural Network (NN).
    """
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test_raw  = datasets.MNIST(root='./data', train=False, download=True)

    x_train_full = mnist_train_raw.data.float() / 255.0
    y_train_full = mnist_train_raw.targets
    x_test       = mnist_test_raw.data.float() / 255.0
    y_test       = mnist_test_raw.targets

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full.numpy(), y_train_full.numpy(), 
        test_size=10000, stratify=y_train_full.numpy(), random_state=42
    )

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val   = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset   = TensorDataset(x_val, y_val)
    test_dataset  = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, x_test, y_test

def load_data_cnn(batch_size):
    
    """
    Loads and prepares data for the CNN with stratified train/val split.
    """

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),   # augmentation only for train
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load raw MNIST (no transform yet)
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    X = mnist_train_raw.data.numpy()
    y = mnist_train_raw.targets.numpy()

    # Stratified split indices
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        test_size=10000,
        stratify=y,
        random_state=42
    )

    # Apply transforms separately
    train_dataset = TensorDataset(
        datasets.MNIST(root='./data', train=True, download=True, transform=train_transform),
        train_idx
    )
    val_dataset = TensorDataset(
        datasets.MNIST(root='./data', train=True, download=True, transform=test_transform),
        val_idx
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
