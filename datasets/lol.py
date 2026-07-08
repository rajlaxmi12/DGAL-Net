"""
==========================================================
DGAL-Net
LOL Dataset
==========================================================
"""

import os

from PIL import Image
from torch.utils.data import Dataset


class LOLDataset(Dataset):
    """
    LOL Dataset

    Folder structure:

    train/
        low/
        high/

    test/
        low/
        high/
    """

    def __init__(
        self,
        root_dir,
        transform=None,
    ):

        self.root_dir = root_dir

        self.transform = transform

        self.low_dir = os.path.join(root_dir, "low")

        self.high_dir = os.path.join(root_dir, "high")

        self.image_names = sorted([

            file

            for file in os.listdir(self.low_dir)

            if file.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                )
            )

        ])

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]

        low_path = os.path.join(
            self.low_dir,
            image_name,
        )

        high_path = os.path.join(
            self.high_dir,
            image_name,
        )

        low = Image.open(
            low_path
        ).convert("RGB")

        high = Image.open(
            high_path
        ).convert("RGB")

        if self.transform is not None:

            low = self.transform(low)

            high = self.transform(high)

        return {

            "low": low,

            "high": high,

            "name": image_name,

        }