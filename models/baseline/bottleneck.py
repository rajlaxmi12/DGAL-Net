"""
==========================================================
DGAL-Net
Baseline Bottleneck
==========================================================
"""

import torch.nn as nn

from models.common.adain import AdaIN
from models.common.residual_block import ResidualBlock


class Bottleneck(nn.Module):
    """
    Bottleneck used in the published IEEE paper.

    Training:
        AdaIN + 3 Residual Blocks

    Inference:
        3 Residual Blocks
    """

    def __init__(self):
        super().__init__()

        self.adain = AdaIN()

        self.rb1 = ResidualBlock(256)
        self.rb2 = ResidualBlock(256)
        self.rb3 = ResidualBlock(256)

    def forward(self,
                feature,
                style_feature=None):

        # AdaIN only during training
        if self.training and style_feature is not None:
            feature = self.adain(feature, style_feature)

        feature = self.rb1(feature)
        feature = self.rb2(feature)
        feature = self.rb3(feature)

        return feature