import math
from datetime import datetime

import torch
import torchvision.transforms as T
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm

from logger import Logger

logger = Logger(show=True).get_logger(__name__)

tensor_to_pil = T.ToPILImage()


class Classifier(nn.Module):
    def __init__(self, model, optim, criterion, device, model_name,
                 print_conf_matrix=False):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = optim
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.print_conf_matrix = print_conf_matrix

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def run_epoch(self, phase, dataloader):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        y_test = []
        y_pred = []
        all_elems_count = 0
        for inputs, labels in tqdm(dataloader):
            bz = inputs.shape[0]
            all_elems_count += bz

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            y_test.extend(labels.cpu())
            y_pred.extend(preds.cpu())
            running_loss += loss.item() * bz
            running_corrects += torch.sum(preds == labels.data)

        conf_matrix = confusion_matrix(y_test, y_pred)
        f1_macro = metrics.f1_score(y_test, y_pred, average="macro")
        epoch_loss = running_loss / all_elems_count
        epoch_acc = running_corrects.double().item() / all_elems_count
        return epoch_loss, epoch_acc, f1_macro, conf_matrix

    def test_epoch(self, dataloader):
        with torch.no_grad():
            return self.run_epoch('test', dataloader)

    def train_epoch(self, dataloader):
        return self.run_epoch('train', dataloader)

    def train_model(self, dataloaders, early_stopping, num_epochs=5):
        logger.info(f"Training model {self.model_name} with params:")
        logger.info(f"Optim: {self.optimizer}")
        logger.info(f"Criterion: {self.criterion}")

        saved_epoch_losses = {'train': [], 'test': []}
        saved_epoch_accuracies = {'train': [], 'test': []}
        saved_epoch_f1_macros = {'train': [], 'test': []}

        # save_by = saved_epoch_losses
        # save_by = saved_epoch_accuracies
        save_by = saved_epoch_f1_macros

        for epoch in range(num_epochs):
            start_time = datetime.now()

            logger.info("=" * 100)
            logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            logger.info('-' * 10)

            for phase in ['train', 'test']:
                logger.info(f"--- Cur phase: {phase}")
                epoch_loss, epoch_acc, f1_macro, conf_matrix = \
                    self.train_epoch(dataloaders[phase]) if phase == 'train' \
                        else self.test_epoch(dataloaders[phase])
                saved_epoch_losses[phase].append(epoch_loss)
                saved_epoch_accuracies[phase].append(epoch_acc)
                saved_epoch_f1_macros[phase].append(f1_macro)
                logger.info(f'{phase} loss: {epoch_loss:.4f}, '
                             f'acc: {epoch_acc:.4f}, f1_macro: {f1_macro:.4f}')
                if self.print_conf_matrix:
                    logger.info("Confusion matrix:")
                    logger.info(conf_matrix)

            end_time = datetime.now()
            epoch_time = (end_time - start_time).total_seconds()
            logger.info("-" * 10)
            logger.info(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60)}")

            early_stopping(save_by['test'][-1], self.model)
            if early_stopping.early_stop:
                logger.info('*** Early stopping ***')
                break
            if f1_macro > 0.95:
                logger.info('*** Needed F1 macro achieved ***')
                break
        logger.info("*** Training Completed ***")
        return self.model

    def test_model(self, dataloader):
        logger.info("*" * 25)
        logger.info(f">> Testing {self.model_name} network")
        epoch_loss, epoch_acc, f1_macro, conf_matrix = self.test_epoch(dataloader)
        logger.info(f"Total test loss: {epoch_loss}")
        logger.info(f"Total test accuracy: {epoch_acc}")
        logger.info(f"Total test F1_macro score: {f1_macro}")
        logger.info("Confusion matrix:")
        logger.info(conf_matrix)
        return epoch_loss, epoch_acc, f1_macro, conf_matrix
