"""
==========================================================
DGAL-Net
Feature Modulation Block (FMB)

Novel Component

This module modulates encoder features according to the
predicted difficulty score.

Input:
    Feature Map      : (B,256,H,W)
    Difficulty Score : (B,1)

Output:
    Modulated Feature: (B,256,H,W)
==========================================================
"""

import torch
import torch.nn as nn


class FeatureModulation(nn.Module):
    """
    Difficulty-Guided Feature Modulation.

    Pipeline

        Difficulty Score
              ↓
         FC(1→64)
              ↓
            ReLU
              ↓
        FC(64→256)
              ↓
          Sigmoid
              ↓
      Channel Attention
              ↓
      Feature × Attention
    """

    def __init__(
        self,
        channels=256,
        hidden_dim=64,
    ):
        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                1,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                channels,
            ),

            nn.Sigmoid(),

        )

    def forward(
        self,
        feature,
        difficulty,
    ):
        """
        Args

            feature:
                (B,C,H,W)

            difficulty:
                (B,1)

        Returns

            modulated_feature:
                (B,C,H,W)
        """

        B, C, H, W = feature.shape

        # ----------------------------------
        # Channel Attention
        # ----------------------------------

        attention = self.attention(
            difficulty
        )

        attention = attention.view(
            B,
            C,
            1,
            1,
        )

        # ----------------------------------
        # Feature Modulation
        # ----------------------------------

        modulated_feature = feature * attention

        return modulated_feature