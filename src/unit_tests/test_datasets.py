import os
import sys
import unittest

import torch

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from dataset_utils import get_FashionMNIST_datasets

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s:`>> %(message)s`',
    handlers=[logging.StreamHandler()]
)


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        logging.info(f"Testing datasets")
        self.train_dataset, self.test_dataset = get_FashionMNIST_datasets(nesting_level='.')
        logging.info(f"Datasets collected")

    def test_datasets_len(self):
        logging.info(f"Testing datasets len...")
        self.assertEqual(len(self.train_dataset), 60000)
        self.assertEqual(len(self.test_dataset), 10000)
        logging.info(f"Testing datasets len passed!")

    def test_datasets_types(self):
        logging.info(f"Testing datasets types...")
        self.assertTrue(isinstance(self.train_dataset.data, torch.Tensor))
        self.assertTrue(isinstance(self.test_dataset.data, torch.Tensor))
        logging.info(f"Testing datasets types passed!")


if __name__ == "__main__":
    unittest.main()
