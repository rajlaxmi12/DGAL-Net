"""
==========================================================
DGAL-Net v2
Residual Block
==========================================================

Residual Block with

• Group Normalization
• Residual Scaling

This improves optimization stability for lightweight
low-light image enhancement networks.
==========================================================
"""

import torch.nn as nn

from models.common.conv_block import ConvBlock


class ResidualBlock(nn.Module):
    """
    Residual Block

    Conv → GN → ReLU
            ↓
       Conv → GN
            ↓
      Residual Scaling
            ↓
       Skip Connection
            ↓
          ReLU
    """

    def __init__(
        self,
        channels,
        residual_scale=0.2,
    ):
        super().__init__()

        self.residual_scale = residual_scale

        # ------------------------------------------
        # First Convolution
        # ------------------------------------------

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ------------------------------------------
        # Second Convolution
        # ------------------------------------------

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=32,
                num_channels=channels,
            ),

        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        # ------------------------------------------
        # Residual Scaling
        # ------------------------------------------

        out = identity + self.residual_scale * out

        out = self.relu(out)

        return out
