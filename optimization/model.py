import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        # Fully connected layer 2
        self.fc2 = nn.Linear(120, n_classes)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolutional layer followed by ReLU and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 32 * 8 * 8)
        # Apply the first fully connected layer followed by ReLU
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer to produce the class scores
        x = self.fc2(x)
        return x
