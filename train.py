"""
==========================================================
DGAL-Net
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

from models.baseline.baseline_net import BaselineNet

from trainer.trainer import Trainer


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def main():

    print("=" * 60)
    print("DGAL-Net")
    print("Baseline Training")
    print("=" * 60)

    # ------------------------------------------
    # Seed
    # ------------------------------------------

    set_seed(config.SEED)

    # ------------------------------------------
    # Data
    # ------------------------------------------

    print("\nLoading Dataset...")

    train_loader = get_train_loader()

    val_loader = get_val_loader()

    print("Train Batches :", len(train_loader))

    print("Validation Batches :", len(val_loader))

    # ------------------------------------------
    # Model
    # ------------------------------------------

    print("\nBuilding Network...")

    model = BaselineNet()

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

    # ------------------------------------------
    # Trainer
    # ------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        device=config.DEVICE,

        lr=config.LR,

    )

    # ------------------------------------------
    # Training
    # ------------------------------------------

    trainer.fit(
        epochs=config.EPOCHS
    )


if __name__ == "__main__":

    main()