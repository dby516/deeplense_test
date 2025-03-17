# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from data_proc import LensDataset
from datetime import datetime
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import *

import wandb

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
batch_size = 64

data_dir = "../Samples"
dataset = LensDataset(data_dir, transform=None)
train_size = int(dataset*0.8)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Hyperparameters
best_val_loss = np.inf
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "cnn", "cnn"))
num_epochs = 100
learning_rate = 1e-3

# Initialize model, optimizer, criterion
model = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Cosine Annealing LR Scheduler (better convergence)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Initialize wandb
wandb.init(
    project="lens-classifier",
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model": "UNet",
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }
)

# Training, testing
best_model = model
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


for epoch in range(num_epochs):
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
    
    # scheduler.step()

    avg_train_loss = running_loss / len(train_loader)
    wandb.log({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        # "learning_rate": scheduler.get_last_lr()[0]
    })
    
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")
    # Validate every 5 epochs
    if (epoch + 1) % 5 == 0:
        v_loss, v_acc, _ = evaluate_model(model, device, criterion, val_loader)
        print(f"Validate: Epoch {epoch+1}/{num_epochs}, Loss: {v_loss}, Accuracy: {v_acc}")

        wandb.log({
            "epoch": epoch+1,
            "val_loss": v_loss,
            "val_accuracy": v_acc
        })
        if v_loss < best_val_loss: # record the model that performs best on validation set
            best_val_loss = v_loss
            best_model = model
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(model.state_dict(), f"{model_path}_{epoch+1}epc_{current_date}.pth") # save model

print("[INFO] Training complete!")

# Evaluation
t_loss, t_acc, _ = evaluate_model(best_model, device, criterion, val_loader)
print(f"Testing Result: Loss: {t_loss}, Accuracy: {t_acc}")

wandb.log({"test_loss": t_loss, "test_accuracy": t_acc})