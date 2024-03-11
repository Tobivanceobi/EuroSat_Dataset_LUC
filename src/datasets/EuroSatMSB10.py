import os
import time

import numpy as np
import rasterio
import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import Dataset

from src.colors import bcolors
from src.pickle_loader import save_object

c = bcolors()


class EuroSatMSB10(Dataset):
    def __init__(self,
                 dataframe,
                 root_dir,
                 x_chan, y_chan,
                 scaler=None,
                 augment=None, transform=None,
                 n_jobs=-4):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.x_chan = x_chan
        self.y_chan = y_chan

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"{c.OKCYAN}Number of images: {len(dataframe)}{c.ENDC}")
        print(f"{c.OKCYAN}Number of jobs:   {n_jobs} {c.ENDC}\n")

        start_time = time.time()
        samples = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(dataframe))
        )
        end_time = time.time()

        t = end_time - start_time
        print(f"\n{c.OKBLUE}Time taken:      {int((t - (t % 60))/ 60)} min {t % 60} sec {c.ENDC}")

        self.samples = torch.tensor(np.array(samples), dtype=torch.float32)

        if scaler:
            print(f"\n{c.OKGREEN}MinMax scaling the images...{c.ENDC}")
            self.min_max_scale()

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])

        with rasterio.open(img_path) as src:
            image = np.array(src.read())

        image = image.astype(np.float32)

        return image

    def min_max_scale(self):
        scaled_tensor = torch.zeros_like(self.samples)
        for i in range(self.samples.shape[1]):  # Loop over channels
            channel = self.samples[:, i, :, :]
            min_val = torch.min(channel)
            max_val = torch.max(channel)
            scaled_tensor[:, i, :, :] = (channel - min_val) / (max_val - min_val)
        self.samples = scaled_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.samples[idx]

        target = data[self.y_chan]
        image = data[self.x_chan]

        if self.augment:
            image = self.augment(image).squeeze(0)

        if self.transform:
            image = self.transform(image)

        return image, target
