import torch

from losses.adaptive_loss import AdaptiveLoss


def main():

    model = AdaptiveLoss()

    difficulty = torch.tensor([
        [0.10],
        [0.50],
        [0.90],
    ])

    weights = model(difficulty)

    print("Difficulty")

    print(difficulty)

    print()

    print("L1 Weight")

    print(weights["l1"])

    print()

    print("Perceptual Weight")

    print(weights["perceptual"])


if __name__ == "__main__":

    main()