import torch

from utils.metrics import ImageMetrics

metrics = ImageMetrics()

prediction = torch.rand(
    2,
    3,
    256,
    256,
)

target = torch.rand(
    2,
    3,
    256,
    256,
)

result = metrics.compute(
    prediction,
    target,
)

print(result)