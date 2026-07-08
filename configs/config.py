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

    # Windows Dataset Path
    DATA_ROOT = "/kaggle/input/datasets/soumikrakshit/lol-dataset/lol_dataset"

    LOL_TRAIN = os.path.join(DATA_ROOT, "our485")
    LOL_VAL = os.path.join(DATA_ROOT, "eval15")
    
    # Optional datasets (if available)
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

    # Keep 2 only for testing.
    # After confirming training works, change to 100.
    EPOCHS = 100

    BATCH_SIZE = 4

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
