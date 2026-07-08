# tests/test_conv_block.py

import torch

from models.common.conv_block import ConvBlock

x = torch.randn(2, 3, 256, 256)

model = ConvBlock(3, 64)

y = model(x)

print(y.shape)

total = sum(p.numel() for p in model.parameters())

print(f"Parameters: {total}")