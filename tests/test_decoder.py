import torch

from models.baseline.decoder import Decoder

model = Decoder()

x = torch.randn(
    2,
    256,
    32,
    32,
)

y = model(x)

print("Input :", x.shape)

print("Output:", y.shape)

params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters:", params)

print("Min:", y.min().item())

print("Max:", y.max().item())