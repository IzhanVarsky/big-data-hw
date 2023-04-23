from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_FashionMNIST_datasets(nesting_level=".."):
    dataset_root = f'{nesting_level}/data'
    train_dataset = datasets.FashionMNIST(
        root=dataset_root,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root=dataset_root,
        train=False,
        download=True,
        transform=ToTensor()
    )
    return train_dataset, test_dataset


def get_FashionMNIST_dataloaders(nesting_level=".."):
    train_dataset, test_dataset = get_FashionMNIST_datasets(nesting_level)

    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataloaders = {
        'train': train_dataloader,
        'test': test_dataloader,
    }
    return dataloaders
