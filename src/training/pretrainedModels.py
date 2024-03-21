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


class EuroSatPreTrainedModel(pl.LightningModule):
    def __init__(self,
                 backbone,
                 dropout,
                 layers,
                 learning_rate,
                 weight_decay,
                 freeze_backbone=True,
                 momentum=0.9,
                 gamma=0.9,
                 n_classes=10):
        super(EuroSatPreTrainedModel, self).__init__()
        self.backbone = backbone
        self.weight_decay = weight_decay
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if self.backbone.__class__.__name__ == "AlexNet":
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        elif self.backbone.__class__.__name__ == "ViTForImageClassification":
            num_features = self.backbone.classifier.out_features
        else:
            num_features = self.backbone.fc.out_features

        self.classifier = nn.Sequential()
        if len(layers) > 0:
            for i, layer in enumerate(layers):
                self.classifier.add_module(f"fc_{i}", nn.Linear(num_features, layer))
                self.classifier.add_module(f"relu_{i}", nn.ReLU())
                self.classifier.add_module(f"dropout_{i}", nn.Dropout(p=dropout))
                num_features = layer
            self.classifier.add_module(f"fc_out", nn.Linear(num_features, n_classes))
        else:
            self.classifier = nn.Linear(num_features, n_classes)

        for param in self.classifier.parameters():
            param.requires_grad = True

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gamma = gamma

        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = 0.0
        self.ep_out = []
        self.ep_true = []

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

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
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        _, labels = torch.max(labels, 1)
        _, predicted = torch.max(outputs, 1)

        loss = self.criterion(outputs, labels)

        self.ep_out.append(predicted.cpu().numpy())
        self.ep_true.append(labels.cpu().numpy())

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_accuracy', self.accuracy, prog_bar=True, on_step=True, on_epoch=True)

    def on_test_epoch_end(self):
        all_preds = np.concatenate(self.ep_out)
        all_true = np.concatenate(self.ep_true)
        correct = np.sum(all_preds == all_true)
        self.accuracy = correct / len(all_true)
        self.log('test_accuracy', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.ep_out = []
        self.ep_true = []

    def configure_optimizers(self):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                              lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]
