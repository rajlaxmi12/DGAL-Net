"""
==========================================================
DGAL-Net v2
Testing Script
==========================================================
"""

import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from configs.config import config
from datasets.dataloader import get_val_loader
from models.dgal.dgal_net import DGALNet
from utils.metrics import ImageMetrics


def load_model(checkpoint_path, device):
    """
    Load trained DGAL-Net model.
    """

    model = DGALNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])

    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def main():

    print("=" * 60)
    print("DGAL-Net v2 Testing")
    print("=" * 60)

    device = config.DEVICE

    model = load_model(
        "checkpoints/best_model.pth",
        device,
    )

    metrics = ImageMetrics(device)

    val_loader = get_val_loader()

    os.makedirs("outputs", exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():

        for batch in tqdm(val_loader):

            low = batch["low"].to(device)
            high = batch["high"].to(device)
            name = batch["name"][0]

            # --------------------------------------
            # Forward
            # --------------------------------------

            output = model(low)

            if isinstance(output, tuple):

                prediction = output[0]

            else:

                prediction = output

            # --------------------------------------
            # Metrics
            # --------------------------------------

            result = metrics.compute(
                prediction,
                high,
            )

            total_psnr += result["psnr"]
            total_ssim += result["ssim"]
            count += 1

            # --------------------------------------
            # Save Output
            # --------------------------------------

            save_image(
                prediction.clamp(0, 1),
                os.path.join("outputs", name),
            )

    print("\n" + "=" * 60)
    print(f"Images Tested : {count}")
    print(f"Average PSNR  : {total_psnr / count:.4f} dB")
    print(f"Average SSIM  : {total_ssim / count:.4f}")
    print("Enhanced images saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
