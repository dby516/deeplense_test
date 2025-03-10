import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from data_proc import LensDataset
from datetime import datetime
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
import torch

def evaluate_model(model, device, criterion, data_loader):
    '''
    Evaluate the model on a test/validation set.

    Returns:
    - avg_loss (float): Average loss across all batches.
    - accuracy (float): Model accuracy.
    - mistakes (list): List of (input, true_label, predicted_label) for incorrect predictions.
    '''
    model.eval()
    tot_loss = 0
    correct = 0
    total = 0
    mistakes = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            tot_loss += loss.item()

            # Get predicted class
            preds = outputs.argmax(dim=1)

            # Calculate accuracy
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # Record mistakes
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    mistakes.append((inputs[i].cpu(), labels[i].cpu().item(), preds[i].cpu().item()))

    # Compute average loss and accuracy
    avg_loss = tot_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy, mistakes


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [8000, len(dataset) - 8000])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
valset = LensDataset("/root/autodl-fs/dataset/val")
val_loader = DataLoader(valset, batch_size=64, shuffle=True)

model_path = "../checkpoints/cnn/"

for epoch in range(20):  # Should overfit within 10 epochs
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Clip gradients to prevent vanishing/exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    # Validate every 5 epochs
    if (epoch + 1) % 5 == 0:
        v_loss, v_acc, _ = evaluate_model(model, device, criterion, val_loader)
        print(f"Validate: Epoch {epoch+1}/{num_epochs}, Loss: {v_loss}, Accuracy: {v_acc}")
        if v_loss < best_val_loss: # record the model that performs best on validation set
            best_val_loss = v_loss
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(model.state_dict(), f"{model_path}cnn_{epoch+1}epc_{current_date}") # save model

print("[INFO] Training complete!")

""" Test """
t_loss, t_acc, _ = evaluate_model(model, device, criterion, test_loader)
print(f"Testing Result: Loss: {t_loss}, Accuracy: {t_acc}")
