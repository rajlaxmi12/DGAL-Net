"""
==========================================================
DGAL-Net
Baseline Decoder
==========================================================
"""

import torch.nn as nn

from models.common.conv_block import ConvBlock


class Decoder(nn.Module):
    """
    Decoder for baseline LLIE network.

    Input:
        (N,256,32,32)

    Output:
        (N,3,128,128)
    """

    def __init__(self):
        super().__init__()

        self.up1 = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.conv1 = ConvBlock(
            256,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up2 = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.conv2 = ConvBlock(
            128,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.final = nn.Conv2d(
            64,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.up1(x)

        x = self.conv1(x)

        x = self.up2(x)

        x = self.conv2(x)

        x = self.final(x)

        x = self.activation(x)

        return x