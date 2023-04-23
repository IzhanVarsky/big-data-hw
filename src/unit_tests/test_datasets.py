import os
import sys
import unittest

import torch

from dataset_utils import get_FashionMNIST_datasets

sys.path.insert(1, os.path.join(os.getcwd(), "src"))


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.train_dataset, self.test_dataset = get_FashionMNIST_datasets(nesting_level='../..')

    def test_datasets_len(self):
        self.assertEqual(len(self.train_dataset), 60000)
        self.assertEqual(len(self.test_dataset), 10000)

    def test_datasets_types(self):
        self.assertTrue(isinstance(self.train_dataset.data, torch.Tensor))
        self.assertTrue(isinstance(self.test_dataset.data, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
