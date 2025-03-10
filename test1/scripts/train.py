import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from data_proc import LensDataset
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "test1")))
from models import *
# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = LensDataset("/root/autodl-fs/dataset/train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define checkpoint path
checkpoint_path = "/root/autodl-fs/checkpoints/lens_vit_checkpoint.pth"

# Initialize
# model = LensViTClassifier().to(device)
model = LensCNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs by 50%


# # Training Loop
# num_epochs = 200
# min_loss = np.inf
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in dataloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
    
#     if (epoch+1)%20 == 0 and running_loss < min_loss:
#          # Save model checkpoint
#         checkpoint = {
#             "epoch": epoch + 1,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "loss": running_loss / len(dataloader),
#         }
#         min_loss = running_loss
#         torch.save(checkpoint, checkpoint_path)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# print("Training Complete!")


small_dataset, _ = torch.utils.data.random_split(dataset, [5000, len(dataset) - 5000])
small_dataloader = DataLoader(small_dataset, batch_size=64, shuffle=True)

for epoch in range(20):  # Should overfit within 10 epochs
    model.train()
    running_loss = 0.0
    
    for images, labels in small_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # ✅ Clip gradients to prevent vanishing/exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: Mean={param.grad.mean().item()}, Std={param.grad.std().item()}")
    #         print(f"{name}: Mean Grad={param.grad.mean().item()}, Std Grad={param.grad.std().item()}")

    #         break  # Print first param only
    
    # scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(small_dataloader)}")

print("Done!")
