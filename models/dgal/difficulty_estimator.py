"""
==========================================================
DGAL-Net v2
Spatial Difficulty Estimation Module (SDEM)
==========================================================

This module estimates the overall enhancement difficulty
from encoder feature maps.

Instead of directly applying Global Average Pooling,
the network first analyzes spatial feature complexity
using lightweight convolutional layers.

Input:
    Feature Map : (B,256,H,W)

Output:
    Difficulty Score : (B,1)

Range:
    [0,1]
==========================================================
"""

import torch
import torch.nn as nn


class DifficultyEstimator(nn.Module):
    """
    Spatial Difficulty Estimation Module (SDEM)

    Feature Map
        │
        ▼
    Conv3×3
        │
    GroupNorm
        │
      ReLU
        │
    Conv3×3
        │
    GroupNorm
        │
      ReLU
        │
    Global Average Pooling
        │
        ▼
    FC (256→64)
        │
      ReLU
        │
    Dropout
        │
    FC (64→1)
        │
    Sigmoid
        │
        ▼
    Difficulty Score
    """

    def __init__(
        self,
        in_channels=256,
        hidden_dim=64,
    ):
        super().__init__()

        # ------------------------------------------
        # Spatial Difficulty Feature Extraction
        # ------------------------------------------

        self.feature_refinement = nn.Sequential(

            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(32, in_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(32, in_channels),

            nn.ReLU(inplace=True),

        )

        # ------------------------------------------
        # Global Pooling
        # ------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ------------------------------------------
        # Difficulty Prediction Head
        # ------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                in_channels,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(
                hidden_dim,
                1,
            ),

            nn.Sigmoid(),

        )

    def forward(self, feature):
        """
        Args
        ----
        feature : Tensor
            Shape : (B,256,H,W)

        Returns
        -------
        difficulty : Tensor
            Shape : (B,1)
        """

        # ------------------------------------------
        # Spatial Feature Analysis
        # ------------------------------------------

        feature = self.feature_refinement(feature)

        # ------------------------------------------
        # Global Feature Vector
        # ------------------------------------------

        pooled = self.global_pool(feature)

        pooled = torch.flatten(
            pooled,
            start_dim=1,
        )

        # ------------------------------------------
        # Difficulty Prediction
        # ------------------------------------------

        difficulty = self.fc(pooled)

        return difficulty
