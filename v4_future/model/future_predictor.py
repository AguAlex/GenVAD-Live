"""U-Net compact: stack cadre t-k..t-1 (canale) -> predicție cadru t."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FutureFrameUNet(nn.Module):
    def __init__(self, n_input_frames: int = 4, base_channels: int = 32):
        super().__init__()
        self.n_input_frames = n_input_frames
        in_ch = 3 * n_input_frames
        b = base_channels

        self.enc1 = ConvBlock(in_ch, b)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)

        self.head = nn.Conv2d(b, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bot = self.bottleneck(self.pool3(e3))

        d3 = self.up3(bot)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.head(d1))

    @staticmethod
    def anomaly_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE pe batch -> scor scalar per sample."""
        return ((pred - target) ** 2).mean(dim=(1, 2, 3))
