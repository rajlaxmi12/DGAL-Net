"""
==========================================================
DGAL-Net
Difficulty Estimation Module (DEM)

Novel Component

This module predicts a learnable image difficulty score
from encoder features.

Input:
    Feature Map : (B, 256, H, W)

Output:
    Difficulty Score : (B, 1)

Range:
    [0, 1]
==========================================================
"""

import torch
import torch.nn as nn


class DifficultyEstimator(nn.Module):
    """
    Difficulty Estimation Module (DEM)

    Pipeline:
        Feature Map
            ↓
        Global Average Pooling
            ↓
        FC (256 → 64)
            ↓
        ReLU
            ↓
        FC (64 → 1)
            ↓
        Sigmoid
            ↓
        Difficulty Score
    """

    def __init__(
        self,
        in_channels=256,
        hidden_dim=64,
    ):
        super().__init__()

        # ------------------------------------------
        # Global Average Pooling
        # ------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------
        # Difficulty Prediction Head
        # ------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                in_channels,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                1,
            ),

            nn.Sigmoid(),

        )

    def forward(self, feature):
        """
        Args:
            feature : (B,256,H,W)

        Returns:
            difficulty : (B,1)
        """

        # ------------------------------------------
        # Global Feature Vector
        # ------------------------------------------

        x = self.global_pool(feature)

        # (B,256,1,1) -> (B,256)
        x = torch.flatten(x, start_dim=1)

        # ------------------------------------------
        # Difficulty Prediction
        # ------------------------------------------

        difficulty = self.fc(x)

        return difficulty