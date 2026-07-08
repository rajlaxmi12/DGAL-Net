from datasets.transforms import DGALTransforms

from datasets.lol import LOLDataset

transform = DGALTransforms().get_train_transform()

dataset = LOLDataset(

    root_dir="./data/LOL/train",

    transform=transform

)

print(len(dataset))

sample = dataset[0]

print(sample["low"].shape)

print(sample["high"].shape)

print(sample["name"])