import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class LensViTClassifier(nn.Module):
    def __init__(self, num_classes=3, image_size=150, patch_size=15,
                 dim=512, depth=6, heads=8, mlp_dim=1024):
        super(LensViTClassifier, self).__init__()
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size
        # Linear projection for patches
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches+1, dim))
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Append class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.transformer(x)

        return self.mlp_head(x[:, 0])