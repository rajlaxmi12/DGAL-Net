"""
==========================================================
DGAL-Net
Image Transformations
==========================================================
"""

from torchvision import transforms


class DGALTransforms:
    """
    Image preprocessing for DGAL-Net.

    Baseline:
        Resize + ToTensor

    Later (DGAL):
        Augmentations can be enabled.
    """

    def __init__(self, image_size=128):

        self.image_size = image_size

        # --------------------------------------------
        # Training
        # --------------------------------------------

        self.train_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

        ])

        # --------------------------------------------
        # Validation
        # --------------------------------------------

        self.val_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

        ])

        # --------------------------------------------
        # Test
        # --------------------------------------------

        self.test_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

        ])

    def get_train_transform(self):
        return self.train_transform

    def get_val_transform(self):
        return self.val_transform

    def get_test_transform(self):
        return self.test_transform