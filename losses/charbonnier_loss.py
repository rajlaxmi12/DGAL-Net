"""
==========================================================
DGAL-Net v2
Charbonnier Loss
==========================================================

A differentiable approximation of the L1 loss widely used
in image restoration.

Formula:

    L = sqrt((x - y)^2 + eps^2)

Advantages
----------
• More robust than L2
• Smoother than L1
• Stable gradients
• Better convergence
==========================================================
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction,
        target,
    ):
        """
        Parameters
        ----------
        prediction : Tensor
            Shape (B,C,H,W)

        target : Tensor
            Shape (B,C,H,W)

        Returns
        -------
        loss : scalar tensor
        """

        diff = prediction - target

        loss = torch.sqrt(
            diff * diff + self.eps ** 2
        )

        return loss.mean()
