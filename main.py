import torch.cuda
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from tqdm import tqdm


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.num_channels = 1
        self.num_classes = 10
        self.in_sz = 28
        self.layers = nn.Sequential(
            nn.Conv2d(self.num_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(self.in_sz * self.in_sz * 16 // 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    bs = 256
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 1

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for data, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    total_item_cnt = 0
    total_correct_cnt = 0
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(test_dataloader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            pred_classes = output.topk(1, dim=1)[1].flatten()
            correct_pred = (pred_classes == label).sum().item()
            total_item_cnt += len(label)
            total_correct_cnt += correct_pred
    print(f"Accuracy: {total_correct_cnt / total_item_cnt}")
