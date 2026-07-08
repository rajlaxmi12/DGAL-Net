import torch

from losses.reconstruction_loss import ReconstructionLoss

criterion = ReconstructionLoss()

pred = torch.randn(2,3,128,128)

gt = torch.randn(2,3,128,128)

loss = criterion(pred, gt)

print(loss)