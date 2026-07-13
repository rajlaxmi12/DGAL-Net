"""
==========================================================
DGAL-Net v2
Training Script
==========================================================
"""

import random
import numpy as np
import torch

from configs.config import config

from datasets.dataloader import (
    get_train_loader,
    get_val_loader,
)

# DGAL Network
from models.dgal.dgal_net import DGALNet

from trainer.trainer import Trainer


# ==========================================================
# Seed
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("DGAL-Net v2")
    print("Difficulty Guided Adaptive Low-Light Enhancement")
    print("=" * 70)

    # --------------------------------------------------
    # Seed
    # --------------------------------------------------

    set_seed(config.SEED)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\nLoading Dataset...")

    train_loader = get_train_loader()

    val_loader = get_val_loader()

    print(f"Train Batches      : {len(train_loader)}")
    print(f"Validation Batches : {len(val_loader)}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nBuilding DGAL-Net...")

    model = DGALNet()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        device=config.DEVICE,

        lr=config.LR,

        epochs=config.EPOCHS,

    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    trainer.fit(

        epochs=config.EPOCHS,

        start_epoch=trainer.start_epoch,

    )

    print("\nTraining Finished Successfully.")


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()
