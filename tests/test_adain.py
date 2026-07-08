import torch

from models.common.adain import AdaIN

content = torch.randn(2,64,128,128)

style = torch.randn(2,64,128,128)

adain = AdaIN()

output = adain(content,style)

print(content.shape)

print(style.shape)

print(output.shape)