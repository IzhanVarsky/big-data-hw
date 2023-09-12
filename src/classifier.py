import math
from datetime import datetime

import torch
import torchvision.transforms as T
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s:`>> %(message)s`',
    handlers=[logging.StreamHandler()]
)

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
        logging.info(f"Training model {self.model_name} with params:")
        logging.info(f"Optim: {self.optimizer}")
        logging.info(f"Criterion: {self.criterion}")

        saved_epoch_losses = {'train': [], 'test': []}
        saved_epoch_accuracies = {'train': [], 'test': []}
        saved_epoch_f1_macros = {'train': [], 'test': []}

        # save_by = saved_epoch_losses
        # save_by = saved_epoch_accuracies
        save_by = saved_epoch_f1_macros

        for epoch in range(num_epochs):
            start_time = datetime.now()

            logging.info("=" * 100)
            logging.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
            logging.info('-' * 10)

            for phase in ['train', 'test']:
                logging.info("--- Cur phase:", phase)
                epoch_loss, epoch_acc, f1_macro, conf_matrix = \
                    self.train_epoch(dataloaders[phase]) if phase == 'train' \
                        else self.test_epoch(dataloaders[phase])
                saved_epoch_losses[phase].append(epoch_loss)
                saved_epoch_accuracies[phase].append(epoch_acc)
                saved_epoch_f1_macros[phase].append(f1_macro)
                logging.info('{} loss: {:.4f}, acc: {:.4f}, f1_macro: {:.4f}'
                      .format(phase, epoch_loss, epoch_acc, f1_macro))
                if self.print_conf_matrix:
                    logging.info("Confusion matrix:")
                    logging.info(conf_matrix)

            end_time = datetime.now()
            epoch_time = (end_time - start_time).total_seconds()
            logging.info("-" * 10)
            logging.info(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60)}")

            early_stopping(save_by['test'][-1], self.model)
            if early_stopping.early_stop:
                logging.info('*** Early stopping ***')
                break
            if f1_macro > 0.95:
                logging.info('*** Needed F1 macro achieved ***')
                break
        logging.info("*** Training Completed ***")
        return self.model

    def test_model(self, dataloader):
        logging.info("*" * 25)
        logging.info(f">> Testing {self.model_name} network")
        epoch_loss, epoch_acc, f1_macro, conf_matrix = self.test_epoch(dataloader)
        logging.info("Total test loss:", epoch_loss)
        logging.info("Total test accuracy:", epoch_acc)
        logging.info("Total test F1_macro score:", f1_macro)
        logging.info("Confusion matrix:")
        logging.info(conf_matrix)
        return epoch_loss, epoch_acc, f1_macro, conf_matrix
