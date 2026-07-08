from models.baseline.baseline_net import BaselineNet
from trainer.optimizer import build_optimizer
from trainer.scheduler import build_scheduler

model = BaselineNet()

optimizer = build_optimizer(model)

scheduler = build_scheduler(optimizer)

print("Initial LR:", optimizer.param_groups[0]["lr"])

for epoch in range(1, 61):

    scheduler.step()

    if epoch % 20 == 0:

        print(
            f"Epoch {epoch}:",
            optimizer.param_groups[0]["lr"]
        )