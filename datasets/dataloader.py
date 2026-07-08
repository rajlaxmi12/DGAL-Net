"""
==========================================================
DGAL-Net
DataLoader Factory
==========================================================
"""

from torch.utils.data import DataLoader

from configs.config import config
from datasets.transforms import DGALTransforms
from datasets.lol import LOLDataset


def get_train_loader():
    """
    Build training dataloader.
    """

    transform = DGALTransforms(
        config.IMAGE_SIZE
    ).get_train_transform()

    dataset = LOLDataset(
        root_dir=config.LOL_TRAIN,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    return loader


def get_val_loader():
    """
    Build validation dataloader.
    """

    transform = DGALTransforms(
        config.IMAGE_SIZE
    ).get_val_transform()

    dataset = LOLDataset(
        root_dir=config.LOL_VAL,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    return loader