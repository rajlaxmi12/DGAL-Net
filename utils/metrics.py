"""
==========================================================
DGAL-Net
Image Quality Metrics
==========================================================
"""

import torch

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


class ImageMetrics:
    """
    Compute image quality metrics.

    Supported:
        - PSNR
        - SSIM
    """

    class ImageMetrics:

    def __init__(self, device):

        self.psnr = PeakSignalNoiseRatio(
            data_range=1.0
        ).to(device)

        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)

    @torch.no_grad()
    def compute(self, prediction, target):

        prediction = prediction.clamp(0, 1)

        target = target.clamp(0, 1)

        psnr = self.psnr(
            prediction,
            target,
        )

        ssim = self.ssim(
            prediction,
            target,
        )

        return {

            "psnr": psnr.item(),

            "ssim": ssim.item(),

        }