import torch

from models.dgal.feature_modulation import FeatureModulation


model = FeatureModulation()

feature = torch.randn(
    2,
    256,
    64,
    64,
)

difficulty = torch.rand(
    2,
    1,
)

output = model(
    feature,
    difficulty,
)

print("Feature Shape    :", feature.shape)
print("Difficulty Shape :", difficulty.shape)
print("Output Shape     :", output.shape)

params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters:", params)