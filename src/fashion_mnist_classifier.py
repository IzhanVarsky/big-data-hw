import torch
from torch import optim, nn

from classifier import Classifier
from model import CNNModel

import yaml

with open("config.yaml") as stream:
    config = yaml.safe_load(stream)

input_img_size = config["model"]["input_img_size"]
num_channels = config["model"]["num_channels"]
num_classes = config["model"]["num_classes"]

lr = float(config["train_params"]["lr"])


def get_FashionMNIST_classifier():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'CNNModel'

    model = CNNModel(input_img_size, num_channels, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    classifier = Classifier(model, optimizer, criterion, device, model_name)
    return classifier
