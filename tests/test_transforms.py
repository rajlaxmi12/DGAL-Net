# test_transforms.py

from PIL import Image
from datasets.transforms import DGALTransforms

transform = DGALTransforms(image_size=256)

img = Image.open("sample.jpg").convert("RGB")

tensor = transform.get_train_transform()(img)

print(tensor.shape)