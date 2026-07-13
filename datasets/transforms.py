"""
==========================================================
DGAL-Net v2
Image Transformations
==========================================================

Training:
    • Resize
    • Random Horizontal Flip
    • Random Vertical Flip
    • Random Rotation
    • Color Jitter
    • ToTensor

Validation/Test:
    • Resize
    • ToTensor
==========================================================
"""

from torchvision import transforms


class DGALTransforms:

    def __init__(self, image_size=128):

        self.image_size = image_size

        # --------------------------------------------------
        # Training Transform
        # --------------------------------------------------

        self.train_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
            ),

            transforms.ColorJitter(

                brightness=0.15,

                contrast=0.15,

                saturation=0.05,

                hue=0.02,

            ),

            transforms.ToTensor(),

        ])

        # --------------------------------------------------
        # Validation Transform
        # --------------------------------------------------

        self.val_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

        ])

        # --------------------------------------------------
        # Test Transform
        # --------------------------------------------------

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
