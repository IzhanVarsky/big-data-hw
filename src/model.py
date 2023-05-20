from torch import nn


class CNNModel(nn.Module):
    def __init__(self, input_img_size=28, num_channels=1, num_classes=10):
        super(CNNModel, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.in_sz = input_img_size
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
