"""
==========================================================
DGAL-Net
Baseline Network
==========================================================
"""

import torch.nn as nn

from models.baseline.encoder import Encoder
from models.baseline.bottleneck import Bottleneck
from models.baseline.decoder import Decoder


class BaselineNet(nn.Module):
    """
    Baseline Low-Light Image Enhancement Network.

    Architecture:
        Input
          ↓
        Encoder
          ↓
      Bottleneck
          ↓
        Decoder
          ↓
      Enhanced Image
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.bottleneck = Bottleneck()

        self.decoder = Decoder()

    def forward(self, x, style_feature=None):

        # Encode
        feature = self.encoder(x)

        # Bottleneck
        feature = self.bottleneck(
            feature,
            style_feature
        )

        # Decode
        output = self.decoder(feature)

        return output
        