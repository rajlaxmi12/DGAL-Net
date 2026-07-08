"""
==========================================================
DGAL-Net
Difficulty Guided Adaptive Low-Light Enhancement Network
==========================================================

Architecture

Input
   │
   ▼
Encoder
   │
   ▼
Difficulty Estimator
   │
   ├──────────────► Difficulty Score
   │
   ▼
Feature Modulation
   │
   ▼
Bottleneck
   │
   ▼
Decoder
   │
   ▼
Enhanced Image

Returns:
    enhanced_image
    difficulty_score
"""

import torch.nn as nn

from models.baseline.encoder import Encoder
from models.baseline.bottleneck import Bottleneck
from models.baseline.decoder import Decoder

from models.dgal.difficulty_estimator import DifficultyEstimator
from models.dgal.feature_modulation import FeatureModulation


class DGALNet(nn.Module):
    """
    DGAL-Net

    Novel LLIE Architecture.

    Returns

        enhanced_image

        difficulty_score
    """

    def __init__(self):

        super().__init__()

        # -----------------------------------------
        # Feature Extraction
        # -----------------------------------------

        self.encoder = Encoder()

        # -----------------------------------------
        # Difficulty Estimation Module
        # -----------------------------------------

        self.difficulty_estimator = DifficultyEstimator()

        # -----------------------------------------
        # Feature Modulation Module
        # -----------------------------------------

        self.feature_modulation = FeatureModulation()

        # -----------------------------------------
        # Bottleneck
        # -----------------------------------------

        self.bottleneck = Bottleneck()

        # -----------------------------------------
        # Decoder
        # -----------------------------------------

        self.decoder = Decoder()

    def forward(
        self,
        x,
        style_feature=None,
    ):

        # -----------------------------------------
        # Encoder
        # -----------------------------------------

        feature = self.encoder(x)

        # -----------------------------------------
        # Difficulty Prediction
        # -----------------------------------------

        difficulty = self.difficulty_estimator(
            feature
        )

        # -----------------------------------------
        # Feature Modulation
        # -----------------------------------------

        feature = self.feature_modulation(
            feature,
            difficulty,
        )

        # -----------------------------------------
        # Bottleneck
        # -----------------------------------------

        feature = self.bottleneck(
            feature,
            style_feature,
        )

        # -----------------------------------------
        # Decoder
        # -----------------------------------------

        output = self.decoder(
            feature
        )

        return output, difficulty