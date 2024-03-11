from typing import Any

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from kornia.metrics import SSIM


class UNetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate, in_channels, weight_decay, out_channels=1, gamma=0.95):
        super(UNetLightningModule, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels
        )
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.weight_decay = weight_decay

        self.criterion = SSIM(window_size=11)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, b10_channel = batch
        b10_pred = self(inputs)
        loss = 1 - self.criterion(b10_pred, b10_channel.unsqueeze(1))
        loss = loss.mean(dim=[2, 3]).sum()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, b10_channel = batch
        b10_pred = self(inputs)
        loss = 1 - self.criterion(b10_pred, b10_channel.unsqueeze(1))
        loss = loss.mean(dim=[2, 3]).sum()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]
