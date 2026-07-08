"""
==========================================================
DGAL-Net
Multi-Scale Perceptual Loss
==========================================================
"""

import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import VGG16_Weights


class VGG16FeatureExtractor(nn.Module):
    """
    Extract features from:

        relu1_2
        relu2_2
        relu3_3
    """

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        )

        features = vgg.features

        self.stage1 = nn.Sequential(*features[:4])
        self.stage2 = nn.Sequential(*features[4:9])
        self.stage3 = nn.Sequential(*features[9:16])

        # Freeze VGG
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, x):

        f1 = self.stage1(x)

        f2 = self.stage2(f1)

        f3 = self.stage3(f2)

        return [f1, f2, f3]


class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.extractor = VGG16FeatureExtractor()

        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction,
        target,
    ):

        # --------------------------------------
        # VERY IMPORTANT
        # Move VGG to same device as input
        # --------------------------------------

        self.extractor = self.extractor.to(
            prediction.device
        )

        prediction_features = self.extractor(
            prediction
        )

        with torch.no_grad():

            target_features = self.extractor(
                target
            )

        loss = 0.0

        for pf, tf in zip(
            prediction_features,
            target_features,
        ):

            loss += self.l1(
                pf,
                tf,
            )

        loss /= len(prediction_features)

        return loss