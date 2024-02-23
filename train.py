import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import accuracy, CNN_MNIST

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 5
BATCH = 1000
LR = 0.005
MODEL_PATH = "./model.pt"


def main():
    train_data = torchvision.datasets.MNIST(
        root="./",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    test_data = torchvision.datasets.MNIST(
        root="./",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH,
        shuffle=True,
        pin_memory=True,
    )
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor) / 255.0
    test_y = test_data.targets
    test_x = test_x.to(device, non_blocking=True)
    test_y = test_y.to(device, non_blocking=True)

    model = CNN_MNIST(device=device).to(device)
    model.summary()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(EPOCH):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Training
                model.train()
                output = model(data)

                train_acc = accuracy(output=output, target=target)
                train_acc_list.append(train_acc)

                train_loss = loss_function(output, target)
                train_loss_list.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # Testing
                with torch.no_grad():
                    model.eval()
                    output = model(test_x)

                    test_acc = accuracy(output=output, target=test_y)
                    test_acc_list.append(test_acc)

                    test_loss = loss_function(output, test_y)
                    test_loss_list.append(test_loss.item())

                tepoch.set_description(f"Epoch {epoch+1}")
                tepoch.set_postfix(
                    train_acc=train_acc,
                    test_acc=test_acc,
                    train_loss=train_loss.item(),
                    test_loss=test_loss.item(),
                )

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)

    # Plot training & testing history
    plt.figure(figsize=(15, 11), dpi=120)
    plt.subplot(221)
    plt.plot(train_acc_list)
    plt.title(f"Training accuracy {train_acc_list[-1]}")
    plt.xlabel("step")

    plt.subplot(222)
    plt.plot(test_acc_list)
    plt.title(f"Testing accuracy {test_acc_list[-1]}")
    plt.xlabel("step")

    plt.subplot(223)
    plt.plot(train_loss_list)
    plt.title(f"Training loss {train_loss_list[-1]}")
    plt.xlabel("step")

    plt.subplot(224)
    plt.plot(test_loss_list)
    plt.title(f"Testing loss {test_loss_list[-1]}")
    plt.xlabel("step")

    plt.savefig("history.png")


if __name__ == "__main__":
    main()
