from torch.utils.data import DataLoader
import pytorch_lightning as pl


class EuroSatDataModule(pl.LightningDataModule):
    def __init__(self, ds_train, ds_val, ds_test, batch_size, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_train = ds_train
        self.dataset_val = ds_val
        self.dataset_test = ds_test

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
