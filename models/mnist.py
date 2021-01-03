"""
MNIST models

model input shape: [batch_size, channel=1, width=28, height=28]

"""
import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
from torchvision import datasets, transforms


def prepare_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_datasets = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_datasets = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = data_utils.DataLoader(train_datasets, shuffle=True, batch_size=200)
    test_loader = data_utils.DataLoader(test_datasets, shuffle=True, batch_size=200)
    return train_loader, test_loader


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(28 * 28, 196),
            nn.Sigmoid(),
            nn.Linear(196, 10),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.model.forward(x)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.model.forward(x)


def train(model, n_epoch):
    net = model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_loader, test_loader = prepare_data_loader()

    iter_count = 0
    losses = list()
    for epoch in range(n_epoch):
        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if iter_count % 100 == 0:
                print(f"{iter_count} iter - loss: {loss.item()}")
                losses.append(loss.item())
            iter_count += 1

        with torch.no_grad():
            loss_sum = 0
            acc_sum = 0
            for data, labels in test_loader:
                output = net(data)
                loss = criterion(output, labels)
                loss_sum += loss.item()
                ans = output.argmax(dim=1, keepdim=True)
                acc_sum += ans.eq(labels.view_as(ans)).sum().item() / data.size(0)

            print(f"{epoch} epoch test - loss: {loss_sum / len(test_loader)} - acc: {acc_sum / len(test_loader)}")

    torch.save(net.state_dict(), f"data/mnist_{model.__name__}.pt")


def to_onnx(model):
    net = model()
    state_dict = torch.load(f"data/mnist_{model.__name__}.pt", map_location=torch.device("cpu"))
    net.load_state_dict(state_dict)
    x = torch.zeros(1, 1, 28, 28)
    for opv in [9, 10, 11, 12]:
        torch.onnx.export(net, x, f"data/mnist_{model.__name__}_v{opv}.onnx", opset_version=opv)


if __name__ == "__main__":
    for m, ne in zip([Net1, Net2], [4, 2]):
        print(m.__name__)
        train(m, ne)
        to_onnx(m)
