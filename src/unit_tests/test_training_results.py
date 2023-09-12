import os
import sys
import unittest

import yaml

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from dataset_utils import get_FashionMNIST_dataloaders
from fashion_mnist_classifier import get_FashionMNIST_classifier

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s:`>> %(message)s`',
    handlers=[logging.StreamHandler()]
)

nesting_level = '.'
with open(f"{nesting_level}/config.yaml") as stream:
    config = yaml.safe_load(stream)

ckpt_path = f'{nesting_level}/{config["ckpt_path"]}'


class TestTrainingResults(unittest.TestCase):
    def setUp(self) -> None:
        logging.info(f"Collecting dataloaders")
        self.dataloaders = get_FashionMNIST_dataloaders(nesting_level=nesting_level)
        logging.info(f"Loading classifier")
        self.classifier = get_FashionMNIST_classifier()
        self.classifier.load_checkpoint(ckpt_path)
        logging.info(f"Checkpoint loaded")

    def test_classifier_metrics(self):
        logging.info(f"Testing classifier metrics...")
        epoch_loss, epoch_acc, f1_macro, conf_matrix = \
            self.classifier.test_model(self.dataloaders['test'])
        self.assertTrue(epoch_acc >= 0.7)
        self.assertTrue(f1_macro >= 0.7)
        logging.info(f"Tests passed!")


if __name__ == "__main__":
    unittest.main()
