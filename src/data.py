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


class AugmentedTensorDataset(TensorDataset):
    """
    A custom TensorDataset that applies a transform to its data.
    """
    def __init__(self, tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        
        # This part of the code is specific to the CNN data pipeline
        if x.ndim == 3 and x.shape[0] == 1:
            # Squeeze the channel dimension to make the tensor compatible with ToPILImage
            x = x.squeeze(0)
        
        # Convert the tensor to a PIL Image for augmentation (e.g., RandomRotation)
        x_pil = transforms.ToPILImage()(x)
        
        # Apply the full transform pipeline
        if self.transform:
            x = self.transform(x_pil)
        
        # Return the transformed tensor and the label
        return x, y

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
    Loads and prepares data for the Convolutional Neural Network (CNN).
    """
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test_raw  = datasets.MNIST(root='./data', train=False, download=True)

    x_train_full = mnist_train_raw.data.unsqueeze(1).float() / 255.0
    y_train_full = mnist_train_raw.targets
    x_test       = mnist_test_raw.data.unsqueeze(1).float() / 255.0
    y_test       = mnist_test_raw.targets

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full.numpy(), y_train_full.numpy(), 
        test_size=10000, stratify=y_train_full.numpy(), random_state=42
    )

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val   = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val, dtype=torch.long)

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = AugmentedTensorDataset((x_train, y_train), transform=train_transform)
    val_dataset   = AugmentedTensorDataset((x_val, y_val), transform=test_transform)
    test_dataset  = AugmentedTensorDataset((x_test, y_test), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_test, y_test