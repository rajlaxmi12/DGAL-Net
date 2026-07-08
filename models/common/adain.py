"""
==========================================================
DGAL-Net
Adaptive Instance Normalization
==========================================================
"""

import torch
import torch.nn as nn


class AdaIN(nn.Module):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, feat):
        """
        Compute channel-wise mean and std

        Input:
            feat -> (N,C,H,W)

        Returns:
            mean -> (N,C,1,1)
            std  -> (N,C,1,1)
        """

        N, C = feat.size()[:2]

        feat = feat.view(N, C, -1)

        mean = feat.mean(dim=2).view(N, C, 1, 1)

        std = feat.std(dim=2).view(N, C, 1, 1) + self.eps

        return mean, std

    def forward(self, content_feat, style_feat):

        content_mean, content_std = self.calc_mean_std(content_feat)

        style_mean, style_std = self.calc_mean_std(style_feat)

        normalized = (content_feat - content_mean) / content_std

        output = normalized * style_std + style_mean

        return output