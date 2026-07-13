"""
==========================================================
DGAL-Net v2
Color Consistency Loss
==========================================================

This loss preserves the global color distribution by
matching the RGB channel statistics between the enhanced
image and the ground-truth image.

Loss:

L_color = ||μ(pred) - μ(gt)||1

where μ is the mean intensity of each RGB channel.

==========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorLoss(nn.Module):
    """
    Channel-wise Color Consistency Loss
    """

    def __init__(self):
        super().__init__()

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
        Scalar Color Loss
        """

        # -----------------------------------------
        # Compute Mean RGB Values
        # -----------------------------------------

        pred_mean = prediction.mean(dim=(2, 3))

        target_mean = target.mean(dim=(2, 3))

        # -----------------------------------------
        # L1 Distance
        # -----------------------------------------

        loss = F.l1_loss(
            pred_mean,
            target_mean,
        )

        return loss
