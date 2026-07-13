"""
==========================================================
DGAL-Net v2
Image Transformations
==========================================================

Deterministic preprocessing for paired low-light image
enhancement.

Since the LOL dataset is a paired dataset, identical
pixel correspondence between the low-light image and
ground-truth image must be preserved.

Therefore, only deterministic transformations are applied.

Training:
    • Resize
    • ToTensor

Validation:
    • Resize
    • ToTensor

Testing:
    • Resize
    • ToTensor
==========================================================
"""

from torchvision import transforms


class DGALTransforms:
    """
    Deterministic preprocessing for paired image restoration.
    """

    def __init__(self, image_size=128):

        self.image_size = image_size

        # --------------------------------------------------
        # Common Transform
        # --------------------------------------------------

        common_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

        ])

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        self.train_transform = common_transform

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        self.val_transform = common_transform

        # --------------------------------------------------
        # Test
        # --------------------------------------------------

        self.test_transform = common_transform

    def get_train_transform(self):
        return self.train_transform

    def get_val_transform(self):
        return self.val_transform

    def get_test_transform(self):
        return self.test_transform
