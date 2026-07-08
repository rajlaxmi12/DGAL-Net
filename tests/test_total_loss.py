import torch

from losses.total_loss import TotalLoss


def main():

    criterion = TotalLoss()

    prediction = torch.rand(
        2,
        3,
        256,
        256,
    )

    target = torch.rand(
        2,
        3,
        256,
        256,
    )

    difficulty = torch.tensor([
        [0.20],
        [0.80],
    ])

    losses = criterion(
        prediction,
        target,
        difficulty,
    )

    print()

    print("Total Loss")

    print(losses["total_loss"])

    print()

    print("L1 Loss")

    print(losses["l1_loss"])

    print()

    print("Perceptual Loss")

    print(losses["perceptual_loss"])

    print()

    print("L1 Weight")

    print(losses["l1_weight"])

    print()

    print("Perceptual Weight")

    print(losses["perceptual_weight"])


if __name__ == "__main__":
    main()