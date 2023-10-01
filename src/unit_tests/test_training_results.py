import argparse
import os
import sys
import unittest
import greenplumpython as gp
import pandas as pd
import yaml
import os

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import db_utils
from dataset_utils import get_FashionMNIST_dataloaders
from fashion_mnist_classifier import get_FashionMNIST_classifier

from logger import Logger

logger = Logger(show=True).get_logger(__name__)

nesting_level = '.'


# with open(f"{nesting_level}/config.yaml") as stream:
#     config = yaml.safe_load(stream)
#
# ckpt_path = f'{nesting_level}/{config["ckpt_path"]}'


class TestTrainingResults(unittest.TestCase):
    def setUp(self) -> None:
        logger.info(f"Collecting dataloaders")
        self.dataloaders = get_FashionMNIST_dataloaders(nesting_level=nesting_level)
        logger.info(f"Loading classifier")
        self.classifier = get_FashionMNIST_classifier()
        self.classifier.load_checkpoint(ckpt_path)
        logger.info(f"Checkpoint loaded")

    def test_classifier_metrics(self):
        logger.info(f"Testing classifier metrics...")
        epoch_loss, epoch_acc, f1_macro, conf_matrix = \
            self.classifier.test_model(self.dataloaders['test'])

        data = pd.DataFrame({
            "epoch_loss": [epoch_loss],
            "epoch_acc": [epoch_acc],
            "f1_macro": [f1_macro],
        })
        db_utils.write_results(db, data)

        self.assertTrue(epoch_acc >= 0.7)
        self.assertTrue(f1_macro >= 0.7)
        logger.info(f"Tests passed!")


if __name__ == "__main__":
    db_credentials = db_utils.get_db_credentials()
    db = gp.database(params=db_credentials)

    t = db_utils.read_db_table(db, table_name=db_utils.TABLE_NAME.model_weights)
    ckpt_path = list(t)[-1]['model_path']

    unittest.main(exit=False)

    logger.info("=" * 50)
    logger.info("Results table:")
    logger.info(db_utils.read_db_table(db, table_name=db_utils.TABLE_NAME.model_results))
    logger.info("-" * 20)
    logger.info("Weights table:")
    logger.info(db_utils.read_db_table(db, table_name=db_utils.TABLE_NAME.model_weights))
