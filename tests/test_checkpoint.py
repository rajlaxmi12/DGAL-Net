import os
import torch

from models.baseline.baseline_net import BaselineNet
from trainer.optimizer import build_optimizer
from trainer.scheduler import build_scheduler
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def main():

    model = BaselineNet()

    optimizer = build_optimizer(model)

    scheduler = build_scheduler(optimizer)

    filepath = "./checkpoints/test_checkpoint.pth"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        loss=0.123,
        filepath=filepath,
    )

    epoch = load_checkpoint(
        filepath,
        model,
        optimizer,
        scheduler,
    )

    print("Loaded Epoch:", epoch)

    print("Checkpoint Exists:",
          os.path.exists(filepath))


if __name__ == "__main__":
    main()