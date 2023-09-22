import argparse
import os
import sys
import unittest
import greenplumpython as gp
import pandas as pd
import yaml

import db_utils

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

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
    parser = argparse.ArgumentParser("Training results testing")
    parser.add_argument("--db-user")
    parser.add_argument("--db-password")
    parser.add_argument("--db-name")
    parser.add_argument("--db-host")
    parser.add_argument("--db-port")

    args = parser.parse_args()

    params = dict(
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name
    )

    db = gp.database(params=params)
    t = db.create_dataframe(table_name="model_weights")
    ckpt_path = list(t)[-1]['model_path']

    unittest.main()
