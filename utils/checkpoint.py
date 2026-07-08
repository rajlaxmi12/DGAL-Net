"""
==========================================================
DGAL-Net
Checkpoint Utilities
==========================================================
"""

import os
import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    loss,
    filepath,
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: optimizer
        scheduler: learning rate scheduler
        epoch: current epoch
        loss: validation loss
        filepath: checkpoint path
    """

    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True,
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "loss": loss,

    }

    torch.save(
        checkpoint,
        filepath,
    )

    print(f"Checkpoint saved -> {filepath}")


def load_checkpoint(
    filepath,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    """
    Load checkpoint.

    Returns:
        epoch
    """

    checkpoint = torch.load(
        filepath,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    print(f"Checkpoint loaded <- {filepath}")

    return checkpoint["epoch"]