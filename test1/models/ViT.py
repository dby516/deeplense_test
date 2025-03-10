import torch
import torch.nn as nn
import torch.nn.functional as F

class LensViTClassifier(nn.Module):
    def __init__(self, num_classes=3, image_size=150, patch_size=15,
                 dim=512, depth=6, heads=8, mlp_dim=1024):
        super(LensViTClassifier, self).__init__()
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        # ✅ Patch embedding (Conv2D) with BatchNorm & LeakyReLU
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.bn_patch = nn.BatchNorm2d(dim)  # Normalize after projection

        # ✅ Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # ✅ Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )

        # ✅ MLP classification head with LeakyReLU & LayerNorm
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.LeakyReLU(0.1),  # Prevent dying neurons
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        # ✅ Patch embedding with BatchNorm & LeakyReLU
        x = self.patch_embed(x)
        x = self.bn_patch(x)  # Normalize patches
        x = F.leaky_relu(x, 0.1)  # Apply LeakyReLU

        x = x.flatten(2).transpose(1, 2)  # Reshape for Transformer

        # ✅ Append class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embed  # Add positional embeddings
        x = self.transformer(x)  # Pass through Transformer

        return self.mlp_head(x[:, 0])  # MLP classifier on CLS token
