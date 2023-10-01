import argparse
import os
import sys
import unittest
import greenplumpython as gp
import pandas as pd
import yaml
import os

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import kafka_utils
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

        data = f'{epoch_loss} {epoch_acc} {f1_macro}'.encode('utf-8')
        producer.send(kafka_utils.PREDICTIONS_TOPIC, data)
        logger.info(f"Producer just sent msg with PREDICTIONS topic")

        self.assertTrue(epoch_acc >= 0.7)
        self.assertTrue(f1_macro >= 0.7)
        logger.info(f"Tests passed!")


if __name__ == "__main__":
    ansible_password = os.environ['ANSIBLE_PASSWORD']
    if ansible_password is None:
        exit(-1)

    kafka_host, kafka_port = kafka_utils.get_kafka_credentials_from_vault(ansible_password)

    kafka_utils.create_topics(kafka_host, kafka_port)

    producer = kafka_utils.get_producer(
        kafka_host=kafka_host,
        kafka_port=kafka_port
    )

    ckpt_consumer = kafka_utils.get_consumer(
        kafka_host=kafka_host,
        kafka_port=kafka_port,
        topic=kafka_utils.CKPT_TOPIC,
    )

    predictions_consumer = kafka_utils.get_consumer(
        kafka_host=kafka_host,
        kafka_port=kafka_port,
        topic=kafka_utils.PREDICTIONS_TOPIC,
    )

    db_credentials = db_utils.get_db_credentials(ansible_password)
    db = gp.database(params=db_credentials)

    t = db_utils.read_db_table(db, table_name=db_utils.TABLE_NAME.model_weights)

    producer.send(kafka_utils.CKPT_TOPIC, list(t)[-1]['model_path'].encode('utf-8'))
    logger.info(f"Producer just sent msg with CKPT topic")

    msg = kafka_utils.get_msg_from_consumer(ckpt_consumer)
    logger.info(f"CKPT Consumer got msg: {msg}")
    ckpt_path = msg.value
    unittest.main()

    msg = kafka_utils.get_msg_from_consumer(predictions_consumer)
    logger.info(f"PREDICTIONS Consumer got msg: {msg}")
    epoch_loss, epoch_acc, f1_macro = msg.value.split()
    data = pd.DataFrame({
        "epoch_loss": [epoch_loss],
        "epoch_acc": [epoch_acc],
        "f1_macro": [f1_macro],
    })
    db_utils.write_results(db, data)

    logger.info("Results table:")
    logger.info(db_utils.read_db_table(db, table_name=db_utils.TABLE_NAME.model_weights))
