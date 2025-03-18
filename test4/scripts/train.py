# Import necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
from datetime import datetime

from data_proc import LensDataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import *  # Now imports correctly

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
batch_size = 64
data_dir = "/home/bingyao/deeplense/test4/Samples"
dataset = LensDataset(data_dir, transform=None)
train_size = int(len(dataset) * 0.8)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Hyperparameters
num_epochs = 100
learning_rate = 1e-4
timesteps = 1000
best_val_loss = np.inf
# Initialize model & optimizer
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Initialize wandb
wandb.init(
    project="lens-diffusion",
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model": "UNet-DDPM",
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "timesteps": timesteps,
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }
)

# Noise scheduler (Linear for DDPM)
def linear_noise_schedule(t, beta_start=1e-4, beta_end=0.02):
    return beta_start + (beta_end - beta_start) * t / timesteps

betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

def add_noise(x, t, noise=None):
    """
    Adds noise to the image x at a specific timestep t using the forward diffusion process.
    """
    if noise is None:
        noise = torch.randn_like(x).to(device)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images in train_loader:
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Add noise to the images
        noisy_images, noise = add_noise(images, t)

        # Predict the noise using the model
        pred_noise = model(noisy_images, t.float())

        # Compute loss (how well the model predicts noise)
        loss = criterion(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Log training loss
    wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss}")

    # Evaluate every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                batch_size = images.shape[0]
                t = torch.randint(0, timesteps, (batch_size,), device=device).long()

                noisy_images, noise = add_noise(images, t)
                pred_noise = model(noisy_images, t.float())
                val_loss += criterion(pred_noise, noise).item()

        avg_val_loss = val_loss / len(val_loader)

        # Log validation loss
        wandb.log({
            "epoch": epoch+1,
            "val_loss": avg_val_loss,
        })

        print(f"Validate: Epoch {epoch+1}/{num_epochs}, Loss: {avg_val_loss}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_save_path = f"/home/bingyao/deeplense/test4/checkpoints/ddpm_{epoch+1}epc_{current_date}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"[INFO] Best model saved at {model_save_path}")

print("[INFO] Training complete!")

# Generate & evaluate images
def generate_samples(model, num_samples=16):
    model.eval()
    samples = torch.randn((num_samples, 1, 64, 64)).to(device)  # Start from random noise

    with torch.no_grad():
        for i in reversed(range(timesteps)):
            t = torch.full((num_samples,), i, device=device).long()
            noise_pred = model(samples, t.float())

            if i > 0:
                beta_t = betas[i]
                alpha_t = alphas[i]
                noise = torch.randn_like(samples) if i > 1 else torch.zeros_like(samples)
                samples = (samples - beta_t / torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t) + torch.sqrt(beta_t) * noise

    return samples

generated_images = generate_samples(model, num_samples=16)

# Compute FID using real vs generated images
from pytorch_fid import fid_score

fid = fid_score.calculate_fid_given_paths(["/home/bingyao/deeplense/test4/Samples", "generated_samples"], 64, device, 2048)
wandb.log({"FID Score": fid})

print(f"FID Score: {fid}")
