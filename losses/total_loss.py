"""
==========================================================
DGAL-Net v2
Adaptive Multi-Objective Total Loss
==========================================================

L_total =
L_charbonnier
+ λ1 L_perceptual
+ λ2 L_gradient
+ λ3 L_color
==========================================================
"""

import torch.nn as nn

from losses.charbonnier_loss import CharbonnierLoss
from losses.perceptual_loss import PerceptualLoss
from losses.gradient_loss import GradientLoss
from losses.color_loss import ColorLoss
from losses.adaptive_loss import AdaptiveLoss


class TotalLoss(nn.Module):

    def __init__(self):
        super().__init__()

        # -----------------------------------------
        # Individual Losses
        # -----------------------------------------

        self.charbonnier_loss = CharbonnierLoss()

        self.perceptual_loss = PerceptualLoss()

        self.gradient_loss = GradientLoss()

        self.color_loss = ColorLoss()

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

        # -----------------------------------------
        # Compute Individual Losses
        # -----------------------------------------

        charbonnier = self.charbonnier_loss(
            prediction,
            target,
        )

        perceptual = self.perceptual_loss(
            prediction,
            target,
        )

        gradient = self.gradient_loss(
            prediction,
            target,
        )

        color = self.color_loss(
            prediction,
            target,
        )

        # -----------------------------------------
        # Adaptive Weights
        # -----------------------------------------

        if difficulty is None:

            charbonnier_weight = 1.0
            perceptual_weight = 0.20
            gradient_weight = 0.05
            color_weight = 0.03

        else:

            weights = self.adaptive_loss(
                difficulty
            )

            charbonnier_weight = weights["charbonnier"]

            perceptual_weight = weights["perceptual"]

            gradient_weight = weights["gradient"]

            color_weight = weights["color"]

        # -----------------------------------------
        # Total Loss
        # -----------------------------------------

        total_loss = (

            charbonnier_weight * charbonnier

            +

            perceptual_weight * perceptual

            +

            gradient_weight * gradient

            +

            color_weight * color

        )

        return {

            "total_loss": total_loss,

            "charbonnier_loss": charbonnier,

            "perceptual_loss": perceptual,

            "gradient_loss": gradient,

            "color_loss": color,

            "charbonnier_weight": charbonnier_weight,

            "perceptual_weight": perceptual_weight,

            "gradient_weight": gradient_weight,

            "color_weight": color_weight,

        }
