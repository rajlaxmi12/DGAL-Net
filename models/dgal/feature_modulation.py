"""
==========================================================
DGAL-Net v2
Residual Difficulty Guided Feature Modulation (RDFM)
==========================================================

This module adaptively modulates encoder features according
to the predicted image difficulty.

Unlike simple channel attention, the original feature
representation is preserved using residual modulation.

Input
-----
Feature Map      : (B,256,H,W)

Difficulty Score : (B,1)

Output
------
Enhanced Feature : (B,256,H,W)
==========================================================
"""

import torch
import torch.nn as nn


class FeatureModulation(nn.Module):
    """
    Residual Difficulty Guided Feature Modulation

    Difficulty
        │
        ▼
      FC
        │
      ReLU
        │
      FC
        │
     Sigmoid
        │
        ▼
 Channel Attention
        │
        ▼
Feature × Attention
        │
Residual Fusion
        │
        ▼
Enhanced Feature
    """

    def __init__(
        self,
        channels=256,
        hidden_dim=64,
        residual_scale=0.2,
    ):

        super().__init__()

        self.residual_scale = residual_scale

        # ------------------------------------------
        # Difficulty Embedding
        # ------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                1,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
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
        Parameters
        ----------
        feature :
            (B,C,H,W)

        difficulty :
            (B,1)

        Returns
        -------
        enhanced_feature :
            (B,C,H,W)
        """

        B, C, H, W = feature.shape

        # ------------------------------------------
        # Generate Channel Attention
        # ------------------------------------------

        attention = self.attention(difficulty)

        attention = attention.view(
            B,
            C,
            1,
            1,
        )

        # ------------------------------------------
        # Difficulty Guided Modulation
        # ------------------------------------------

        modulated_feature = feature * attention

        # ------------------------------------------
        # Residual Feature Fusion
        # ------------------------------------------

        enhanced_feature = feature + (
            self.residual_scale * modulated_feature
        )

        return enhanced_feature
