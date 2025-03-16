import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout(0.2) # Reduce overfitting

    def forward(self, x):
        identity = x  # Identity connection remains unchanged
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return F.relu(out)

class ResCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ResCNN, self).__init__()

        # Initial Conv layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Intermediate Convs for Channel Expansion
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(128)

        # Residual Blocks (keeping channels constant)
        self.layer1 = ResidualBlock(16, kernel_size=5)
        self.layer2 = ResidualBlock(32, kernel_size=3)
        self.layer3 = ResidualBlock(64, kernel_size=3)
        # self.layer4 = ResidualBlock(128, kernel_size=3)

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.layer2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.layer3(x)

        # x = F.relu(self.bn4(self.conv4(x)))
        # x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

