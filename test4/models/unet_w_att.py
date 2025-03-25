import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.size()
        query = self.query(x).view(batch, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H*W)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, -1, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)

        return self.gamma * out + x

class Bottleneck(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.resblock = ResidualBlock(channels, channels, time_emb_dim)
        self.attention = SelfAttention(channels)

    def forward(self, x, t_emb):
        x = self.resblock(x, t_emb)
        x = self.attention(x)
        return x



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

        # Ensure residual connection only applies when in_channels != out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        x_res = self.residual(x)
        x = F.silu(self.bn1(self.conv1(x)) + time_emb)
        x = self.dropout(self.bn2(self.conv2(x)))
        return x + x_res

class UNetAtt(nn.Module):
    def __init__(self, img_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Apply Downsampling AFTER each ResidualBlock
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(img_channels, base_channels, time_emb_dim),  
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim),  
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),  
        ])
        
        self.downsample = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # 64 → 128
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # 128 → 256
        ])

        # self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.bottleneck = Bottleneck(base_channels * 4, time_emb_dim)


        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim),
            ResidualBlock(base_channels, base_channels, time_emb_dim),
        ])

        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256 → 128
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128 → 64
        ])

        self.final_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t.unsqueeze(-1))

        skips = []
        for i, (resblock, down) in enumerate(zip(self.encoder_blocks, self.downsample + [None])):
            x = resblock(x, t_emb)
            skips.append(x)
            if down:
                x = down(x)  # Downsampling **after** resblock

        x = self.bottleneck(x, t_emb)

        for i, (resblock, up, skip) in enumerate(zip(self.decoder_blocks, self.upsample + [None], reversed(skips))):
            # Ensure skip connections match spatial dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

            x = x + skip
            x = resblock(x, t_emb)
            if up:
                x = up(x)

        return self.final_conv(x)
