"""
==========================================================
DGAL-Net
Training Logger
==========================================================
"""

import os
from datetime import datetime


class TrainingLogger:
    """
    Console + File Logger.
    """

    def __init__(self, log_dir="logs"):

        self.best_loss = float("inf")

        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, "training_log.txt")

    def log(
        self,
        epoch,
        train_loss,
        val_loss,
        psnr,
        ssim,
        lr,
    ):

        is_best = val_loss < self.best_loss

        if is_best:
            self.best_loss = val_loss

        print("=" * 60)
        print(f"Epoch {epoch}")
        print(f"Train Loss : {train_loss:.6f}")
        print(f"Val Loss   : {val_loss:.6f}")
        print(f"LR         : {lr:.6f}")
        print(f"PSNR       : {psnr:.4f}")
        print(f"SSIM       : {ssim:.4f}")
        print("Best Model :", "YES" if is_best else "NO")
        print("=" * 60)

        with open(self.log_file, "a") as f:
            f.write(
                f"{datetime.now()} | "
                f"Epoch={epoch} | "
                f"Train={train_loss:.6f} | "
                f"Val={val_loss:.6f} | "
                f"PSNR={psnr:.4f} | "
                f"SSIM={ssim:.4f} | "
                f"LR={lr:.6f}\n"
            )