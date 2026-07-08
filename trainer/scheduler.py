"""
==========================================================
DGAL-Net
Learning Rate Scheduler
==========================================================
"""

from torch.optim.lr_scheduler import StepLR


def build_scheduler(
    optimizer,
    step_size=20,
    gamma=0.5,
):
    """
    StepLR scheduler used in the baseline paper.

    Args:
        optimizer : Optimizer
        step_size : Decay every N epochs
        gamma     : LR decay factor

    Returns:
        scheduler
    """

    scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    return scheduler