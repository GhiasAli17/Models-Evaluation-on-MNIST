import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF

def augment_and_duplicate(x, y, rotation_deg=10):
    """
    Apply rotation augmentation once and duplicate dataset.
    Handles both [N, H, W] (NN case) and [N, C, H, W] (CNN case).
    """
    x_out_list = []
    for img in x:
        # ensure shape [C, H, W]
        if img.ndim == 2:  # [H, W]
            img = img.unsqueeze(0)  # -> [1, H, W]

        angle = np.random.uniform(-rotation_deg, rotation_deg)
        img_rot = TF.rotate(img, angle)

        x_out_list.append(img_rot)

    x_rot = torch.stack(x_out_list)

    # remove channel if original had none
    if x.ndim == 3:  # [N, H, W]
        x_rot = x_rot.squeeze(1)  # remove channel dim

    # duplicate labels
    y_rot = y.clone() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    # concatenate
    x_out = torch.cat([x, x_rot], dim=0)
    y_out = torch.cat([y, y_rot], dim=0)

    return x_out, y_out


def load_data_nn(batch_size):
    """
    Loads and prepares data for the Neural Network (NN).
    Duplicates training set with 10% rotation augmentation.
    """
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test_raw  = datasets.MNIST(root='./data', train=False, download=True)

    # these are already tensors
    x_train_full = mnist_train_raw.data.float() / 255.0
    y_train_full = mnist_train_raw.targets
    x_test       = mnist_test_raw.data.float() / 255.0
    y_test       = mnist_test_raw.targets

    # split: outputs numpy arrays
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full.numpy(), y_train_full.numpy(),
        test_size=10000, stratify=y_train_full.numpy(), random_state=42
    )

    # convert numpy -> tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val   = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val, dtype=torch.long)

    # test set was already tensor -> just cast dtype
    x_test = x_test.to(torch.float32)
    y_test = y_test.to(torch.long)

    # --- augment + duplicate ---
    x_train, y_train = augment_and_duplicate(x_train, y_train)

    # flatten for NN
    x_train_flat = x_train.view(len(x_train), -1)
    x_val_flat   = x_val.view(len(x_val), -1)
    x_test_flat  = x_test.view(len(x_test), -1)

    # datasets
    train_dataset = TensorDataset(x_train_flat, y_train)
    val_dataset   = TensorDataset(x_val_flat, y_val)
    test_dataset  = TensorDataset(x_test_flat, y_test)

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_test_flat, y_test


def load_data_cnn(batch_size):
    """
    Loads and prepares data for CNN.
    Doubles training set with rotation augmentation.
    """
    base_transform = transforms.ToTensor()
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)

    X = mnist_train_raw.data.numpy()
    y = mnist_train_raw.targets.numpy()

    # split indices
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        test_size=10000,
        stratify=y,
        random_state=42
    )

    # base + rotated datasets
    mnist_train_base = datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    mnist_train_rot  = datasets.MNIST(root='./data', train=True, download=True,
                                      transform=transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]))

    # concat for doubling
    train_dataset = ConcatDataset([
        Subset(mnist_train_base, train_idx),
        Subset(mnist_train_rot, train_idx)
    ])

    # validation (no rotation)
    mnist_val_tf = datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    val_dataset  = Subset(mnist_val_tf, val_idx)

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
