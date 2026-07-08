"""
==========================================================
DGAL-Net
L1 Reconstruction Loss
==========================================================
"""

import torch.nn as nn


class L1Loss(nn.Module):
    """
    Pixel-wise L1 Reconstruction Loss.

    Input
    -----
    prediction : (N,3,H,W)

    target : (N,3,H,W)

    Output
    ------
    scalar loss
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
        self,
        prediction,
        target,
    ):

        return self.loss(
            prediction,
            target,
        )