import torch

from models.baseline.encoder import Encoder

model = Encoder()

x = torch.randn(2,3,128,128)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters:", params)