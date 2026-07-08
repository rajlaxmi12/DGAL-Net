from models.baseline.baseline_net import BaselineNet
from trainer.optimizer import build_optimizer

model = BaselineNet()

optimizer = build_optimizer(model)

print(optimizer)