import torch

from losses.perceptual_loss import PerceptualLoss

criterion = PerceptualLoss()

prediction = torch.rand(
    2,
    3,
    128,
    128
)

target = torch.rand(
    2,
    3,
    128,
    128
)

loss = criterion(
    prediction,
    target
)

print("Perceptual Loss:", loss.item())