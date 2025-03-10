"""
CNN Classifier

Bingyao, 03-08-2025
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
        
        # ✅ Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # ✅ BatchNorm after conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # ✅ BatchNorm after conv2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # ✅ BatchNorm after conv3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # ✅ BatchNorm after conv4

        self.pool = nn.MaxPool2d(2, 2)
        
        # ✅ Fully Connected layers with BatchNorm
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)  # ✅ BatchNorm after fc1
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)  # ✅ BatchNorm after fc2
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))  # ✅ BatchNorm after fc1
        x = F.relu(self.bn_fc2(self.fc2(x)))  # ✅ BatchNorm after fc2
        x = self.fc3(x)  # No activation here (CrossEntropyLoss expects raw logits)

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