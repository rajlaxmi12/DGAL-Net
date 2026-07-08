"""
==========================================================
DGAL-Net
Optimizer Builder
==========================================================
"""

import torch.optim as optim


def build_optimizer(model, lr=2e-4):
    """
    Build Adam optimizer exactly as used in the baseline paper.

    Args:
        model: PyTorch model
        lr: learning rate

    Returns:
        optimizer
    """

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
    )

    return optimizer