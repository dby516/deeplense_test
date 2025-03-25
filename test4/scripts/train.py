import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from pytorch_msssim import ssim
import wandb
from datetime import datetime
import os
import sys
from data_proc import LensDataset
from models import UNet, UNetAtt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset & DataLoader
batch_size = 64
data_dir = "/home/bingyao/deeplense/test4/Samples"
dataset = LensDataset(data_dir, transform=None)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
num_epochs = 100
learning_rate = 1e-4
timesteps = 1000
best_val_loss = float('inf')

# Model, optimizer, scheduler
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Loss functions
mse_loss_fn = nn.MSELoss()

def combined_loss(output, target, alpha=0.64):
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)
    mse_loss = mse_loss_fn(output, target)
    return alpha * mse_loss + (1 - alpha) * ssim_loss

# WandB initialization
wandb.init(
    project="lens-diffusion",
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model": "UNet-DDPM",
        "optimizer": "Adam",
        "loss_function": "MSE+SSIM",
        # "scheduler": "CosineAnnealingLR",
        "timesteps": timesteps,
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
    }
)

# Noise scheduler (DDPM)
betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

def add_noise(x, t):
    noise = torch.randn_like(x).to(device)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0

    for images in train_loader:
        images = images.to(device)
        batch_size_current = images.size(0)

        t = torch.randint(0, timesteps, (batch_size_current,), device=device).long()
        noisy_images, noise = add_noise(images, t)

        pred_noise = model(noisy_images, t.float())
        loss = combined_loss(pred_noise, noise)
        # loss = criterion(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # scheduler.step()
    avg_train_loss = train_loss / len(train_loader)
    wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

    print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    # Validation
    if epoch % 5 == 0:
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                batch_size_current = images.size(0)
                t = torch.randint(0, timesteps, (batch_size_current,), device=device).long()
                noisy_images, noise = add_noise(images, t)

                pred_noise = model(noisy_images, t.float())
                val_loss += combined_loss(pred_noise, noise).item()
                # val_loss += criterion(pred_noise, noise).item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"epoch": epoch, "val_loss": avg_val_loss, "learning_rate": optimizer.param_groups[0]['lr']})

        print(f"Epoch [{epoch}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_save_path = f"/home/bingyao/deeplense/test4/checkpoints/att/ddpm_{epoch}epc_{timestamp}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")

print("Training complete!")