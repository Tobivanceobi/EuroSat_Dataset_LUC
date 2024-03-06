import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch import optim, nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.colors import bcolors

c = bcolors()


class EuroSatResNet(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, gamma=0.9):
        super(EuroSatResNet, self).__init__()

        self.num_classes = num_classes
        self.gamma = gamma

        # Model architecture
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.freeze_pretrained_model()

        num_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(num_feats, num_classes)

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = 0

        self.ep_out = []
        self.ep_true = []

    def unfreez_children_layers(self, n_layers):
        children = list(self.model.children())
        children.reverse()
        children = children[1:n_layers + 1]

        print(f"{c.OKGREEN}Unfreezing {n_layers} layers...{c.ENDC}")

        for child in children:
            print(f"\n{c.OKCYAN} {child} {c.ENDC}")
            for param in child.parameters():
                param.requires_grad = True

    def freeze_pretrained_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        _, labels = torch.max(labels, 1)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        _, labels = torch.max(labels, 1)
        _, predicted = torch.max(outputs, 1)

        loss = self.criterion(outputs, labels)

        self.ep_out.append(predicted.cpu().numpy())
        self.ep_true.append(labels.cpu().numpy())

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        all_preds = np.concatenate(self.ep_out)
        all_true = np.concatenate(self.ep_true)
        correct = np.sum(all_preds == all_true)
        self.accuracy = correct / len(all_true)
        self.ep_out = []
        self.ep_true = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma)  # Define the scheduler

        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Specify that the scheduler should step after each epoch
                "frequency": 1,  # Specify that the scheduler should step every epoch
            },
        }

    def adjust_learning_rate(self, new_lr):
        self.learning_rate = new_lr
        self.configure_optimizers()

