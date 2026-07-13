"""
==========================================================
DGAL-Net v2
Difficulty Guided Adaptive Low-Light Enhancement Network
==========================================================

Pipeline

Input Image
      │
      ▼
Encoder
      │
      ▼
Spatial Difficulty Estimation Module (SDEM)
      │
      ├──────────────► Difficulty Score
      │
      ▼
Residual Difficulty Guided Feature Modulation (RDFM)
      │
      ▼
Bottleneck
(AdaIN + Residual Blocks)
      │
      ▼
Decoder
      │
      ▼
Enhanced Image

Returns
-------
enhanced_image
difficulty_score
==========================================================
"""

import torch.nn as nn

from models.baseline.encoder import Encoder
from models.baseline.bottleneck import Bottleneck
from models.baseline.decoder import Decoder

from models.dgal.difficulty_estimator import DifficultyEstimator
from models.dgal.feature_modulation import FeatureModulation


class DGALNet(nn.Module):
    """
    DGAL-Net v2

    Difficulty Guided Adaptive Low-Light Enhancement Network
    """

    def __init__(self):
        super().__init__()

        # ------------------------------------------
        # Feature Extraction
        # ------------------------------------------
        self.encoder = Encoder()

        # ------------------------------------------
        # Spatial Difficulty Estimation
        # ------------------------------------------
        self.difficulty_estimator = DifficultyEstimator()

        # ------------------------------------------
        # Difficulty Guided Feature Modulation
        # ------------------------------------------
        self.feature_modulation = FeatureModulation()

        # ------------------------------------------
        # Bottleneck
        # AdaIN + Residual Blocks
        # ------------------------------------------
        self.bottleneck = Bottleneck()

        # ------------------------------------------
        # Decoder
        # ------------------------------------------
        self.decoder = Decoder()

    def forward(
        self,
        x,
        style_feature=None,
    ):
        # ------------------------------------------
        # Feature Extraction
        # ------------------------------------------
        feature = self.encoder(x)

        # ------------------------------------------
        # Predict Enhancement Difficulty
        # ------------------------------------------
        difficulty_score = self.difficulty_estimator(
            feature
        )

        # ------------------------------------------
        # Adaptive Feature Modulation
        # ------------------------------------------
        feature = self.feature_modulation(
            feature,
            difficulty_score,
        )

        # ------------------------------------------
        # Bottleneck
        # ------------------------------------------
        feature = self.bottleneck(
            feature,
            style_feature,
        )

        # ------------------------------------------
        # Image Reconstruction
        # ------------------------------------------
        output = self.decoder(feature)

        return output, difficulty_score
