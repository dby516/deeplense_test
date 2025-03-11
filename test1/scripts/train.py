import torch
import torch.nn as nn
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
# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "train"))
val_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "val"))

dataset = LensDataset(train_path)
valset = LensDataset(val_path)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [24000, len(dataset) - 24000])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(valset, batch_size=64, shuffle=True)

# Hyperparameters
best_val_loss = np.inf
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "cnn", "cnn"))
num_epochs = 100
learning_rate = 1e-3

# Initialization
model = LensCNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Cosine Annealing LR Scheduler (better convergence)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
# scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, total_steps=num_epochs * len(train_loader))

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
    
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    # Validate every 5 epochs
    if (epoch + 1) % 5 == 0:
        v_loss, v_acc, _ = evaluate_model(model, device, criterion, val_loader)
        print(f"Validate: Epoch {epoch+1}/{num_epochs}, Loss: {v_loss}, Accuracy: {v_acc}")
        if v_loss < best_val_loss: # record the model that performs best on validation set
            best_val_loss = v_loss
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(model.state_dict(), f"{model_path}_{epoch+1}epc_{current_date}.pth") # save model

print("[INFO] Training complete!")

""" Test """
t_loss, t_acc, _ = evaluate_model(model, device, criterion, test_loader)
print(f"Testing Result: Loss: {t_loss}, Accuracy: {t_acc}")
