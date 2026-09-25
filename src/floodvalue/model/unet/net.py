"""Compact U-Net (no pretrained weights, no downloads). ~1.9M parameters at base width 16."""

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4 down/up levels; input H, W must be divisible by 16 (predict pads)."""

    def __init__(self, in_channels: int, base: int = 16, dropout: float = 0.1):
        super().__init__()
        w = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList([_block(in_channels, w[0])] +
                                   [_block(w[i], w[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2)
                                  for i in reversed(range(4))])
        self.dec = nn.ModuleList([_block(w[i] * 2, w[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else self.pool(x))
            skips.append(x)
        x = self.drop(skips.pop())
        for up, dec in zip(self.ups, self.dec):
            x = up(x)
            x = dec(torch.cat([skips.pop(), x], dim=1))
        return self.head(x)[:, 0]  # logits (N, H, W)
