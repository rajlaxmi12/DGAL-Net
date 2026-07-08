"""
==========================================================
DGAL-Net
Total Loss
==========================================================

Total Loss for DGAL-Net

Baseline:
    Total = L1 + λ * Perceptual

DGAL-Net:
    Difficulty
          │
          ▼
    Adaptive Loss
          │
          ▼
Dynamic Weights
          │
          ▼
L1 + Perceptual
"""

import torch.nn as nn

from losses.l1_loss import L1Loss
from losses.perceptual_loss import PerceptualLoss
from losses.adaptive_loss import AdaptiveLoss


class TotalLoss(nn.Module):
    """
    Total Loss for DGAL-Net.
    """

    def __init__(self):
        super().__init__()

        # -----------------------------------------
        # Individual Losses
        # -----------------------------------------

        self.l1_loss = L1Loss()

        self.perceptual_loss = PerceptualLoss()

        # -----------------------------------------
        # Adaptive Weight Generator
        # -----------------------------------------

        self.adaptive_loss = AdaptiveLoss()

    def forward(
        self,
        prediction,
        target,
        difficulty=None,
    ):
        """
        Parameters
        ----------
        prediction : Tensor

        target : Tensor

        difficulty : Tensor or None
            Shape (B,1)

        Returns
        -------
        dict
        """

        # -----------------------------------------
        # Compute Individual Losses
        # -----------------------------------------

        l1 = self.l1_loss(
            prediction,
            target,
        )

        perceptual = self.perceptual_loss(
            prediction,
            target,
        )

        # -----------------------------------------
        # Baseline Weights
        # -----------------------------------------

        if difficulty is None:

            l1_weight = 1.0

            perceptual_weight = 0.2

        else:

            weights = self.adaptive_loss(
                difficulty
            )

            l1_weight = weights["l1"]

            perceptual_weight = weights["perceptual"]

        # -----------------------------------------
        # Total Loss
        # -----------------------------------------

        total_loss = (

            l1_weight * l1

            +

            perceptual_weight * perceptual

        )

        return {

            "total_loss": total_loss,

            "l1_loss": l1,

            "perceptual_loss": perceptual,

            "l1_weight": l1_weight,

            "perceptual_weight": perceptual_weight,

        }