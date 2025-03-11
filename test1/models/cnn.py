"""
CNN Classifier

Bingyao, 03-08-2025
"""
"""
Best:
self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3)  # Increased channels, stride added
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)  # More channels, stride added
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        
        self.fc1 = nn.Linear(128, num_classes)

 x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  
        x = self.fc1(x)
        return x
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_proc import LensDataset


import torch
import torch.nn as nn
import torch.nn.functional as F

class LensCNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(LensCNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  
        x = self.fc1(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNClassifier, self).__init__()

        # ✅ More pooling in earlier layers to reduce overfitting
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(4, 4)  # ✅ Earlier downsampling

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(4, 4)  # ✅ More aggressive pooling

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(3, 3)  # ✅ Reduce feature map size further

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(3, 3)  # ✅ Ensures small feature maps before FC

        # ✅ Smaller Fully Connected Layer to reduce memorization
        self.fc1 = nn.Linear(128, num_classes)  # ✅ No hidden FC layer

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)  # ✅ Smaller flattened size reduces FC overfitting
        x = self.fc1(x)

        return x

    

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.pool = nn.MaxPool2d(3, 3)
        self.dropout = nn.Dropout(0.3)  # Dropout added

        self.fc1 = nn.Linear(20000, num_classes)
        # self.bn_fc1 = nn.BatchNorm1d(128)
        # self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        # x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))  # Dropout applied here
        x = self.fc1(x)
        return x
    

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = LensDataset("/root/autodl-fs/dataset/train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize
    model = LensCNNClassifier().to(device)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss() # Use Cross Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

    print("Training Complete!")