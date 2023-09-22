import os
import sys
import unittest

import torch

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from dataset_utils import get_FashionMNIST_datasets

from logger import Logger

logger = Logger(show=True).get_logger(__name__)


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        logger.info(f"Testing datasets")
        self.train_dataset, self.test_dataset = get_FashionMNIST_datasets(nesting_level='.')
        logger.info(f"Datasets collected")

    def test_datasets_len(self):
        logger.info(f"Testing datasets len...")
        self.assertEqual(len(self.train_dataset), 60000)
        self.assertEqual(len(self.test_dataset), 10000)
        logger.info(f"Testing datasets len passed!")

    def test_datasets_types(self):
        logger.info(f"Testing datasets types...")
        self.assertTrue(isinstance(self.train_dataset.data, torch.Tensor))
        self.assertTrue(isinstance(self.test_dataset.data, torch.Tensor))
        logger.info(f"Testing datasets types passed!")


if __name__ == "__main__":
    unittest.main()
