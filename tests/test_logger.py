from utils.logger import TrainingLogger


logger = TrainingLogger()

logger.log_epoch(
    epoch=1,
    total_epochs=200,
    train_loss=0.312,
    val_loss=0.287,
    lr=2e-4,
)

logger.log_epoch(
    epoch=2,
    total_epochs=200,
    train_loss=0.241,
    val_loss=0.295,
    lr=2e-4,
)