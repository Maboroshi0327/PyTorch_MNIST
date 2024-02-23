import torch
import torch.nn as nn
from torchsummary import summary


def accuracy(output, target) -> float:
    output = torch.argmax(output, dim=1)
    acc = (output == target).sum().item() / target.size()[0]
    return acc


class CNN_MNIST(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=7, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=7, out_channels=5, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.drop1 = nn.Dropout(p=0.2)

        self.fc1 = nn.Sequential(
            nn.Linear(320, 150),
            nn.ReLU(),
        )

        self.drop2 = nn.Dropout(p=0.2)

        self.out = nn.Sequential(
            nn.Linear(150, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.out(x)
        return x

    def summary(self):
        summary(model=self, input_size=(1, 28, 28), device=self.device)


if __name__ == "__main__":
    model = CNN_MNIST(device="cpu")
    model.summary()
