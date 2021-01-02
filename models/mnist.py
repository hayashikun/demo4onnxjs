import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
from torchvision import datasets, transforms


def prepare_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_datasets = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_datasets = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = data_utils.DataLoader(train_datasets, shuffle=True, batch_size=100)
    test_loader = data_utils.DataLoader(test_datasets, shuffle=True, batch_size=100)
    return train_loader, test_loader


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.model.forward(x)


def train():
    net = Net1()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    train_loader, test_loader = prepare_data_loader()

    iter_count = 0
    losses = list()
    for epoch in range(5):
        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if iter_count % 1000 == 0:
                print(f"{iter_count} iter - loss: {loss.item()}")
                losses.append(loss.item())
            iter_count += 1

        with torch.no_grad():
            test_loss = list()
            for data, labels in test_loader:
                output = net(data)
                loss = criterion(output, labels)
                test_loss.append(loss.item())
            print(f"{epoch} epoch test - loss: {np.mean(test_loss)}")

    torch.save(net.state_dict(), "data/mnist_1.pt")


def to_onnx():
    net = Net1()
    state_dict = torch.load("data/mnist_1.pt", map_location=torch.device("cpu"))
    net.load_state_dict(state_dict)
    x = torch.zeros(1, 28 * 28)
    for opv in [9, 10, 11, 12]:
        torch.onnx.export(net, x, f"data/mnist_1_v{opv}.onnx", opset_version=opv)


if __name__ == "__main__":
    train()
    to_onnx()
