"""
==========================================================
DGAL-Net v2
Adaptive Multi-Objective Loss Weight Generator
==========================================================

This module dynamically adjusts the loss weights according
to the estimated image difficulty.

Easy Images
-----------
Higher Reconstruction
Lower Perceptual

Hard Images
-----------
Higher Perceptual
Higher Gradient
Higher Color Consistency
==========================================================
"""

import torch
import torch.nn as nn


class AdaptiveLoss(nn.Module):
    """
    Adaptive Multi-Objective Loss Weight Generator
    """

    def __init__(
        self,
        base_charbonnier=1.0,
        base_perceptual=0.20,
        max_perceptual=0.50,
        base_gradient=0.05,
        max_gradient=0.15,
        base_color=0.03,
        max_color=0.08,
    ):
        super().__init__()

        self.base_charbonnier = base_charbonnier

        self.base_perceptual = base_perceptual
        self.max_perceptual = max_perceptual

        self.base_gradient = base_gradient
        self.max_gradient = max_gradient

        self.base_color = base_color
        self.max_color = max_color

    def forward(self, difficulty):

        # ------------------------------------------
        # Clamp Difficulty
        # ------------------------------------------

        difficulty = difficulty.clamp(0.0, 1.0)

        # ------------------------------------------
        # Mini-batch Difficulty
        # ------------------------------------------

        difficulty = difficulty.mean()

        # Slight smoothing for stable optimization
        difficulty = 0.9 * difficulty + 0.1

        # ------------------------------------------
        # Fixed Reconstruction Weight
        # ------------------------------------------

        charbonnier_weight = torch.tensor(
            self.base_charbonnier,
            device=difficulty.device,
            dtype=difficulty.dtype,
        )

        # ------------------------------------------
        # Adaptive Perceptual
        # ------------------------------------------

        perceptual_weight = (
            self.base_perceptual
            + (self.max_perceptual - self.base_perceptual)
            * difficulty
        )

        # ------------------------------------------
        # Adaptive Gradient
        # ------------------------------------------

        gradient_weight = (
            self.base_gradient
            + (self.max_gradient - self.base_gradient)
            * difficulty
        )

        # ------------------------------------------
        # Adaptive Color
        # ------------------------------------------

        color_weight = (
            self.base_color
            + (self.max_color - self.base_color)
            * difficulty
        )

        return {

            "charbonnier": charbonnier_weight,

            "perceptual": perceptual_weight,

            "gradient": gradient_weight,

            "color": color_weight,

        }
        
