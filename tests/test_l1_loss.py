import torch

from losses.l1_loss import L1ReconstructionLoss


criterion = L1ReconstructionLoss()

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

print("Loss:", loss.item())