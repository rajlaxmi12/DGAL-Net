"""
=========================================================
L1 Reconstruction Loss
=========================================================
"""

import torch.nn as nn


class ReconstructionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, prediction, target):

        return self.loss(prediction, target)