"""
==========================================================
DGAL-Net
Residual Block
==========================================================
"""

import torch.nn as nn

from models.common.conv_block import ConvBlock


class ResidualBlock(nn.Module):
    """
    Standard ResNet-style Residual Block

    Conv-BN-ReLU
        ↓
    Conv-BN
        ↓
    Skip Connection
        ↓
    ReLU
    """

    def __init__(self, channels):
        super().__init__()

        # First convolution
        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Second convolution (NO ReLU after this)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = out + identity

        out = self.relu(out)

        return out