# ============================================================
# DGAL-Net Configuration
# IEEE LLIE Project
# ============================================================

import os
import torch

class Config:

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    DATA_ROOT = "/content/drive/MyDrive/data"

    LOL_TRAIN = os.path.join(DATA_ROOT, "LOL/train")
    # Use the official LOL test set as validation for now
    LOL_VAL = os.path.join(DATA_ROOT, "LOL/test")

    LOLV2 = os.path.join(DATA_ROOT, "LOL-v2")
    EXDARK = os.path.join(DATA_ROOT, "ExDark")
    LSRW = os.path.join(DATA_ROOT, "LSRW")

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    IMAGE_SIZE = 256

    CHANNELS = 3

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    EPOCHS = 2

    BATCH_SIZE = 4      # CPU Friendly

    NUM_WORKERS = 0

    PIN_MEMORY = False

    SHUFFLE = True

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    LR = 1e-4

    WEIGHT_DECAY = 1e-5

    BETAS = (0.9, 0.999)

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    STEP_SIZE = 50

    GAMMA = 0.5

    # --------------------------------------------------------
    # Loss Weights
    # --------------------------------------------------------

    LAMBDA_RECON = 1.0

    LAMBDA_PERCEPTUAL = 0.2

    LAMBDA_EDGE = 0.1

    LAMBDA_COLOR = 0.05

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    CHECKPOINT_DIR = "./checkpoints"

    SAVE_EVERY = 10

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    LOG_DIR = "./logs"

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    SEED = 42

config = Config()