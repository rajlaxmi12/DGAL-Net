"""
==========================================================
DGAL-Net
Adaptive Loss Weight Generator
==========================================================

This module dynamically computes loss weights based on the
predicted image difficulty.

Input
-----
difficulty : (B,1)

Output
------
{
    "l1": scalar tensor,
    "perceptual": scalar tensor,
}

The L1 reconstruction weight remains fixed while the
perceptual loss weight increases with image difficulty.
"""

import torch
import torch.nn as nn


class AdaptiveLoss(nn.Module):
    """
    Adaptive Loss Weight Generator.

    Easy Image
        ↓
    Lower perceptual weight

    Difficult Image
        ↓
    Higher perceptual weight
    """

    def __init__(
        self,
        base_l1=1.0,
        base_perceptual=0.20,
        max_perceptual=0.50,
    ):
        super().__init__()

        self.base_l1 = base_l1
        self.base_perceptual = base_perceptual
        self.max_perceptual = max_perceptual

    def forward(self, difficulty):
        """
        Parameters
        ----------
        difficulty : Tensor
            Shape : (B,1)

        Returns
        -------
        dict
            {
                "l1": scalar tensor,
                "perceptual": scalar tensor,
            }
        """

        # ------------------------------------------
        # Clamp difficulty for numerical stability
        # ------------------------------------------

        difficulty = difficulty.clamp(0.0, 1.0)

        # ------------------------------------------
        # Average difficulty of current mini-batch
        # ------------------------------------------

        difficulty = difficulty.mean()

        # ------------------------------------------
        # Fixed reconstruction weight
        # ------------------------------------------

        l1_weight = torch.tensor(
            self.base_l1,
            device=difficulty.device,
            dtype=difficulty.dtype,
        )

        # ------------------------------------------
        # Adaptive perceptual weight
        # ------------------------------------------

        perceptual_weight = (
            self.base_perceptual
            + (self.max_perceptual - self.base_perceptual)
            * difficulty
        )

        return {
            "l1": l1_weight,
            "perceptual": perceptual_weight,
        }