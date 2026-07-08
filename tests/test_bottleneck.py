import torch

from models.baseline.bottleneck import Bottleneck

model = Bottleneck()

# Training mode
model.train()

feature = torch.randn(2,256,32,32)
style = torch.randn(2,256,32,32)

output = model(feature, style)

print("Training")
print(output.shape)

# Inference mode
model.eval()

output = model(feature)

print("Inference")
print(output.shape)

params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters:", params)