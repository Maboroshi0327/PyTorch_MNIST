import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot


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
                in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=5, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.drop1 = nn.Dropout(p=0.5)

        self.fc1 = nn.Sequential(
            nn.Linear(147, 50),
            nn.ReLU(),
        )

        self.drop2 = nn.Dropout(p=0.5)

        self.out = nn.Sequential(
            nn.Linear(50, 10),
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
    # Summary
    model = CNN_MNIST(device="cpu")
    model.summary()

    # Export the model to ONNX format
    torch_input = torch.randn(1, 1, 28, 28).cpu()
    onnx_program = torch.onnx.export(
        model,
        torch_input,
        "model_architecture.onnx",
    )

    # TorchViz Graphviz
    out = model(torch_input)
    g = make_dot(out)
    g.view(filename="model_architecture", cleanup=True)
