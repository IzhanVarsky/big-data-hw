import torch
from torch import optim, nn

from classifier import Classifier
from model import CNNModel


def get_FashionMNIST_classifier():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'CNNModel'

    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    classifier = Classifier(model, optimizer, criterion, device, model_name)
    return classifier
