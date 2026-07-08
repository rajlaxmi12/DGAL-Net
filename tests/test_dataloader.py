from datasets.dataloader import (
    get_train_loader,
    get_val_loader,
)


def main():

    train_loader = get_train_loader()

    print("Number of training batches:", len(train_loader))

    batch = next(iter(train_loader))

    print("Keys:", batch.keys())
    print("Low :", batch["low"].shape)
    print("High:", batch["high"].shape)
    print("Name:", batch["name"][0])

    print()

    val_loader = get_val_loader()

    batch = next(iter(val_loader))

    print("Validation")

    print("Low :", batch["low"].shape)
    print("High:", batch["high"].shape)
    print("Name:", batch["name"][0])


if __name__ == "__main__":
    main()