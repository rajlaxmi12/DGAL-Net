"""
==========================================================
DGAL-Net
Common Convolution Block
==========================================================
Conv → BatchNorm → ReLU
(Baseline IEEE Implementation)
==========================================================
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard Conv-BN-ReLU block.

    Args:
        in_channels (int)
        out_channels (int)
        kernel_size (int)
        stride (int)
        padding (int)

    Input:
        (N,C,H,W)

    Output:
        (N,C_out,H_out,W_out)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)