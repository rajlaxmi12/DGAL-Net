"""
==========================================================
DGAL-Net v2
Common Convolution Block
==========================================================

Conv → GroupNorm → ReLU

Why GroupNorm?

• Stable for small batch sizes
• Better feature normalization
• Improved domain generalization
• Suitable for image restoration
==========================================================
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard Conv-GN-ReLU Block.

    Args
    ----
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
    padding : int

    Input
    -----
    (N,C,H,W)

    Output
    ------
    (N,C_out,H_out,W_out)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=32,
    ):
        super().__init__()

        # Ensure valid number of groups
        groups = min(groups, out_channels)

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=groups,
                num_channels=out_channels,
            ),

            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.block(x)
