import torch

from models.dgal.dgal_net import DGALNet


def main():

    model = DGALNet()

    model.eval()

    x = torch.randn(
        2,
        3,
        256,
        256,
    )

    with torch.no_grad():

        output, difficulty = model(x)

    print()

    print("Input Shape")

    print(x.shape)

    print()

    print("Enhanced Image")

    print(output.shape)

    print()

    print("Difficulty")

    print(difficulty)

    print()

    params = sum(

        p.numel()

        for p in model.parameters()

    )

    print("Total Parameters:", params)


if __name__ == "__main__":

    main()