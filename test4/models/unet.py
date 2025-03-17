import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # Expand to match spatial dimensions
        x_res = self.residual(x)
        x = F.silu(self.bn1(self.conv1(x)) + time_emb)
        x = self.dropout(self.bn2(self.conv2(x)))
        return x + x_res

class UNet(nn.Module):
    def __init__(self, img_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_emb_dim),  # Time step embedding
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.encoder = nn.ModuleList([
            ResidualBlock(img_channels, base_channels, time_emb_dim),
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim),
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim),
        ])
        
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.decoder = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim),
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim),
            ResidualBlock(base_channels, img_channels, time_emb_dim),
        ])

        self.downsample = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
        ])
        
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])

        self.final_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t.unsqueeze(-1))  # Time step embedding

        skips = []
        for layer, down in zip(self.encoder, self.downsample + [None]):
            x = layer(x, t_emb)
            skips.append(x)
            if down:
                x = down(x)

        x = self.bottleneck(x, t_emb)

        for layer, up, skip in zip(self.decoder, self.upsample + [None], reversed(skips)):
            x = up(x) if up else x
            x = x + skip  # Skip connection
            x = layer(x, t_emb)

        return self.final_conv(x)
