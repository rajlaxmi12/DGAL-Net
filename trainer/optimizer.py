"""
==========================================================
DGAL-Net v2
Optimizer Builder
==========================================================

Uses AdamW optimizer for improved optimization stability
and better generalization.

==========================================================
"""

import torch.optim as optim


def build_optimizer(
    model,
    lr=2e-4,
    weight_decay=1e-4,
):
    """
    Build AdamW optimizer.

    Args
    ----
    model : PyTorch model

    lr : float
        Learning rate

    weight_decay : float
        Weight decay coefficient

    Returns
    -------
    optimizer
    """

    optimizer = optim.AdamW(

        model.parameters(),

        lr=lr,

        betas=(0.9, 0.999),

        eps=1e-8,

        weight_decay=weight_decay,

    )

    return optimizer
