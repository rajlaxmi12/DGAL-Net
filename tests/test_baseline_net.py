import torch

from models.baseline.baseline_net import BaselineNet

model = BaselineNet()

# -------------------------------
# Training Mode
# -------------------------------

model.train()

input_image = torch.randn(
    2,
    3,
    128,
    128,
)

style_feature = torch.randn(
    2,
    256,
    32,
    32,
)

output = model(
    input_image,
    style_feature
)

print("===== Training =====")
print("Input :", input_image.shape)
print("Output:", output.shape)

# -------------------------------
# Inference
# -------------------------------

model.eval()

output = model(input_image)

print()

print("===== Inference =====")
print("Output:", output.shape)

params = sum(
    p.numel()
    for p in model.parameters()
)

print()

print("Total Parameters:", params)