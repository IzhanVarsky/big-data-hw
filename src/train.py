from dataset_utils import get_FashionMNIST_dataloaders
from fashion_mnist_classifier import get_FashionMNIST_classifier
from utils import EarlyStopping

from logger import Logger

logger = Logger(show=True).get_logger(__name__)


def run_train():
    dataloaders = get_FashionMNIST_dataloaders()
    dataset_name = "FashionMNIST"

    classifier = get_FashionMNIST_classifier()

    checkpoint_name = f"{classifier.model_name}_{dataset_name}"
    early_stopping = EarlyStopping(model_name=checkpoint_name, save_best=True,
                                   use_early_stop=False, metric_decreasing=False)

    num_epochs = 2
    classifier.train_model(dataloaders, early_stopping, num_epochs=num_epochs)

    logger.info('=' * 50)
    logger.info('Testing LAST model on TRAIN dataset...')
    classifier.test_model(dataloaders['train'])
    logger.info('Testing LAST model on TEST dataset...')
    classifier.test_model(dataloaders['test'])

    logger.info('=' * 50)
    logger.info('Loading and testing the BEST model...')
    classifier.load_checkpoint(early_stopping.get_checkpoint_name())
    logger.info('Testing on TRAIN dataset...')
    classifier.test_model(dataloaders['train'])
    logger.info('Testing on TEST dataset...')
    classifier.test_model(dataloaders['test'])


if __name__ == "__main__":
    run_train()
