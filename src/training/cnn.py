import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl


class EuroSatCNN(nn.Module):
    def __init__(self, num_classes, num_channels, kernel_size):
        super(EuroSatCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=kernel_size, padding=2),  # Output: 32x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x32x32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=2),  # Output: 64x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x16x16
        )
        test_input = torch.randn(1, num_channels, 64, 64)
        b, c, w, h = self.conv_layer(test_input).size()
        print(b, c, w, h)

        self.fc_layer = nn.Sequential(
            nn.Flatten(),  # Flatten the output of conv layers
            nn.Linear(c * w * h, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Assuming 10 classes
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


class LitEuroSatCnn(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, num_channels=12, kernel_size=3):
        super(LitEuroSatCnn, self).__init__()
        self.model = EuroSatCNN(num_classes, num_channels, kernel_size)
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.ep_out = []
        self.ep_true = []

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

    def on_train_epoch_end(self):
        all_preds = np.concatenate(self.ep_out)
        all_true = np.concatenate(self.ep_true)
        correct = np.sum(all_preds == all_true)
        accuracy = correct / len(all_true)
        self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.ep_out = []
        self.ep_true = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
