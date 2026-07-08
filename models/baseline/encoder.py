"""
==========================================================
DGAL-Net
Baseline Encoder
==========================================================
"""

import torch
import torch.nn as nn

from models.common.conv_block import ConvBlock


class Encoder(nn.Module):
    """
    Baseline encoder from the previous IEEE paper.

    Input:
        (N,3,128,128)

    Output:
        (N,256,32,32)
    """

    def __init__(self):
        super().__init__()

        # Stage 1
        self.layer1 = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Stage 2
        self.layer2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Stage 3
        self.layer3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        return x