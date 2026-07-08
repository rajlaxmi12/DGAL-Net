import torch

from models.baseline.baseline_net import BaselineNet

model = BaselineNet()

model.train()

x = torch.randn(2,3,256,256)

style = torch.randn(2,256,64,64)

y = model(x,style)

print("Input :",x.shape)

print("Output:",y.shape)

params = sum(

    p.numel()

    for p in model.parameters()

)

print("Parameters:",params)