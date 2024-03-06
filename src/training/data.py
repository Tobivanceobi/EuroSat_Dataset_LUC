import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from src.datasets.EuroSatMS import EuroSatMS
from src.datasets.EuroSatTest import EuroSatTestSet


class EuroSatDataModule(pl.LightningDataModule):
    N_WORKERS = 10
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, ds_train, ds_val, train_dir, test_dir, batch_size=256):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

        self.dataset_train = ds_train
        self.dataset_val = ds_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.N_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.N_WORKERS)
