"""
==========================================================
DGAL-Net
Trainer
==========================================================
"""

import os
import torch

from trainer.optimizer import build_optimizer
from trainer.scheduler import build_scheduler

from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

from losses.total_loss import TotalLoss

from utils.logger import TrainingLogger
from utils.metrics import ImageMetrics


class Trainer:
    """
    Trainer for DGAL-Net / BaselineNet.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device=None,
        lr=1e-4,
    ):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.model = model.to(self.device)

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        self.train_loader = train_loader
        self.val_loader = val_loader

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        self.criterion = TotalLoss().to(self.device)

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = build_optimizer(
            self.model,
            lr=lr,
        )

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        self.scheduler = build_scheduler(
            self.optimizer
        )

        # --------------------------------------------------
        # Logger
        # --------------------------------------------------

        self.logger = TrainingLogger()

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        self.metrics = ImageMetrics(self.device)

        # --------------------------------------------------
        # Best Validation Loss
        # --------------------------------------------------

        self.best_loss = float("inf")

        # --------------------------------------------------
        # Resume Training
        # --------------------------------------------------

        self.start_epoch = 1

        latest_checkpoint = "checkpoints/latest_checkpoint.pth"

        if os.path.exists(latest_checkpoint):

            print()
            print("=" * 60)
            print("Found previous checkpoint.")
            print("Resuming training...")
            print("=" * 60)

            checkpoint = load_checkpoint(
                filepath=latest_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )

            self.start_epoch = checkpoint["epoch"] + 1

            self.best_loss = checkpoint["loss"]

            print(f"Resume Epoch : {self.start_epoch}")
            print(f"Best Loss    : {self.best_loss:.6f}")
            print("=" * 60)

        print("=" * 60)
        print("DGAL-Net Trainer Initialized")
        print("=" * 60)
        print("Device      :", self.device)
        print("Optimizer   : Adam")
        print("LR          :", lr)
        print("Scheduler   : StepLR")
        print("=" * 60)

    # ======================================================
    # Train One Epoch
    # ======================================================

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        for batch in self.train_loader:

            low = batch["low"].to(self.device)

            high = batch["high"].to(self.device)

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            model_output = self.model(low)

            if isinstance(model_output, tuple):

                prediction, difficulty = model_output

            else:

                prediction = model_output

                difficulty = None

            losses = self.criterion(
                prediction,
                high,
                difficulty,
            )

            total_loss = losses["total_loss"]

            # ------------------------------------------
            # Backpropagation
            # ------------------------------------------

            self.optimizer.zero_grad()

            total_loss.backward()

            self.optimizer.step()

            running_loss += total_loss.item()

        average_loss = running_loss / len(self.train_loader)

        return average_loss

    # ======================================================
    # Validation
    # ======================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        psnr_scores = []

        ssim_scores = []

        for batch in self.val_loader:

            low = batch["low"].to(self.device)

            high = batch["high"].to(self.device)

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            model_output = self.model(low)

            if isinstance(model_output, tuple):

                prediction, difficulty = model_output

            else:

                prediction = model_output

                difficulty = None

            losses = self.criterion(
                prediction,
                high,
                difficulty,
            )

            running_loss += losses["total_loss"].item()

            metrics = self.metrics.compute(
                prediction,
                high,
            )

            psnr_scores.append(metrics["psnr"])

            ssim_scores.append(metrics["ssim"])

        average_loss = running_loss / len(self.val_loader)

        average_psnr = sum(psnr_scores) / len(psnr_scores)

        average_ssim = sum(ssim_scores) / len(ssim_scores)

        return average_loss, average_psnr, average_ssim

    # ======================================================
    # Epoch
    # ======================================================

    def fit_one_epoch(
        self,
        epoch,
        total_epochs,
    ):

        train_loss = self.train_one_epoch()

        val_loss, psnr, ssim = self.validate()

        self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]

        is_best = val_loss < self.best_loss

        if is_best:

            self.best_loss = val_loss

            os.makedirs(
                "checkpoints",
                exist_ok=True,
            )

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=val_loss,
                filepath="checkpoints/best_model.pth",
            )

        # ------------------------------------------
        # Console Log
        # ------------------------------------------

        print()

        print("=" * 60)

        print(f"Epoch [{epoch}/{total_epochs}]")

        print(f"Train Loss : {train_loss:.6f}")

        print(f"Val Loss   : {val_loss:.6f}")

        print(f"LR         : {current_lr:.6f}")

        print("Best Model :", "YES" if is_best else "NO")

        print("=" * 60)

        print(f"PSNR : {psnr:.4f}")

        print(f"SSIM : {ssim:.4f}")

        # ------------------------------------------
        # File Logger
        # ------------------------------------------

        self.logger.log(

            epoch=epoch,

            train_loss=train_loss,

            val_loss=val_loss,

            psnr=psnr,

            ssim=ssim,

            lr=current_lr,

        )

        return train_loss, val_loss

    # ======================================================
    # Full Training Loop
    # ======================================================

    def fit(self, epochs, start_epoch=1):

        print()
        print("Starting Training...")

        for epoch in range(start_epoch, epochs + 1):

            self.fit_one_epoch(
                epoch=epoch,
                total_epochs=epochs,
            )

        print()
        print("Training Completed Successfully!")