import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Classifier, self).__init__()

        # Load pretrained ResNet-50 model
        self.resnet = models.resnet50(weights=None)  # Training from scratch

        # Modify input layer (ResNet expects 3-channel RGB, we have 1-channel grayscale)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final fully connected layer
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ✅ Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Initialize model and move to device
model = ResNet50Classifier(num_classes=3).to(device)
print(model)
