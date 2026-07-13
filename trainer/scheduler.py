"""
==========================================================
DGAL-Net v2
Learning Rate Scheduler
==========================================================

Uses Cosine Annealing Learning Rate Scheduler for
smooth optimization.

Advantages
----------
• Smooth learning rate decay
• Better convergence
• Improved training stability
==========================================================
"""

from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(
    optimizer,
    epochs=100,
    eta_min=1e-6,
):
    """
    Build Cosine Annealing LR Scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    epochs : int
        Total training epochs

    eta_min : float
        Minimum learning rate

    Returns
    -------
    scheduler
    """

    scheduler = CosineAnnealingLR(

        optimizer,

        T_max=epochs,

        eta_min=eta_min,

    )

    return scheduler
