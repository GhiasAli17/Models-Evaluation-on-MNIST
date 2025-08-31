import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (NN) for image classification.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for image classification.
    """
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*14*14, 124),
            nn.ReLU(),
            nn.Linear(124, 10)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x