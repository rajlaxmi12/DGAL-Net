import torch

from models.common.residual_block import ResidualBlock

x = torch.randn(2, 64, 256, 256)

model = ResidualBlock(64)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

params = sum(p.numel() for p in model.parameters())

print(f"Parameters: {params}")