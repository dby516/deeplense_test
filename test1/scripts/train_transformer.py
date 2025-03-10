import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_proc import LensDataset
import sys
import os
import numpy as np
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "test1")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import *
import transformers
import torchvision.transforms as transforms

# Prepare data for ViT
# Feature extractor ensures that images are preprocessed in a way that matches the format used in ViT
feature_extractor = transformers.ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ✅ Convert 1-channel → 3-channel
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

def evaluate_model(model, device, criterion, data_loader):
    '''
    Evaluation function, test the model on testing/validation set
    output:
    - avg_loss, accuracy: average loss and accuracy
    - mistakes: mistakenly predicted samples
    '''
    model.eval() # set model to evaluation mode
    tot_loss = 0
    # to compute accuracy, record prediction result
    correct = 0
    total = 0
    mistakes = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.flatten()
            
            ''' For ResNet & VGG16 '''
            # outputs = model(inputs)
            # tot_loss += criterion(outputs.flatten(), labels).item()
            ''' For ViT '''
            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            tot_loss += criterion(logits, labels.long()).item()
            # Calculate accuracy
            # pred = (outputs > 0.5).float()  # binary predicted results
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            # # Record error prediction
            # if pred != labels:
            #     mistakes.append((inputs.cpu(), labels.cpu().item(), pred.item()))
            
    avg_loss = tot_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, mistakes


# Training function
def train_model(model, device, criterion, optimizer, train_loader, val_loader, epochs, val_gap, model_path):
    ''' 
    Training function
    input:
    - model: model to train
    - device, criteriion, optimizer
    - train_loader: training data
    - val_loader: validation data
    - epochs: number of epochs
    - val_gap: perform validation every val_gap iterations
    - model_path: path to save the best model
    output:
    - val_loss: sequence of average validation loss
    - val_acc: sequence of validation accuracy
    '''
    val_loss = []
    val_acc = []
    best_val_loss = float('inf')

    for epc in range(epochs):
        # Set model to training mode
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            # Send data to device
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.flatten()

            optimizer.zero_grad()

            # Forward pass
            ''' For ResNet & VGG16 '''
            # outputs = model(inputs)
            ''' For ViT '''
            outputs = model(pixel_values=inputs)
            
            # Compute training loss
            ''' For ResNet & VGG16 '''
            # loss = criterion(outputs.flatten(), labels)
            ''' For ViT '''
            logits = outputs.logits
            loss = criterion(logits, labels.long()) # Cross entropy loss requires long input


            # Back propagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print training loss each epoch
        print(f"Train: Epoch {epc+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

        # Validate every val_gap epochs
        if (epc + 1) % val_gap == 0:
            v_loss, v_acc, _ = evaluate_model(model, device, criterion, val_loader)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            print(f"Validate: Epoch {epc+1}/{epochs}, Loss: {v_loss}, Accuracy: {v_acc}")
            if v_loss < best_val_loss: # record the model that performs best on validation set
                best_val_loss = v_loss
                torch.save(model.state_dict(), model_path) # save model

    return val_loss, val_acc



# Load dataset
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "train"))
val_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "val"))
dataset_train = LensDataset(root_dir=train_path, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
dataset_val = LensDataset(root_dir=val_path, transform=transform)
dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=True)


# Load pretrained ViT
model = transformers.ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=3,
    id2label={0: 'no', 1: 'sphere', 2: 'vort'},
    label2id={'no': 0, 'sphere': 1, 'vort': 2}
)


# new_conv_layer = nn.Conv2d(
#     in_channels=1,  # ✅ Change from 3 → 1
#     out_channels=model.vit.embeddings.patch_embeddings.projection.out_channels,
#     kernel_size=model.vit.embeddings.patch_embeddings.projection.kernel_size,
#     stride=model.vit.embeddings.patch_embeddings.projection.stride,
#     padding=model.vit.embeddings.patch_embeddings.projection.padding,
#     bias=model.vit.embeddings.patch_embeddings.projection.bias is not None
# )

# # Copy Weights: Average RGB Weights → 1-channel Weights
# new_conv_layer.weight.data = model.vit.embeddings.patch_embeddings.projection.weight.mean(dim=1, keepdim=True)

# # Replace Patch Embedding Layer
# model.vit.embeddings.patch_embeddings.projection = new_conv_layer

# print("Modified ViT to accept grayscale (1-channel) images.")

# Freeze all parameters except the classifier layer
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False  # Freeze the layer
    else:
        param.requires_grad = True   # Unfreeze the classifier layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Hyperparameters
learning_rate = 1e-3
num_epoch = 10
val_gap = 1 # validate every val_gap iterations
batch_size = 32
model_path = 'models/ViT.pth'

# Initialize loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate) # only modify params in fc


# Train the ViT model with pretrained params
val_loss, val_acc = train_model(model, device, criterion, optimizer, dataloader_train, dataloader_val, num_epoch, val_gap, model_path)