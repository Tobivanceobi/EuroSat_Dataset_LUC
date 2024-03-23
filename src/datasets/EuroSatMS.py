import os
import sys
import time

import numpy as np
import rasterio
import torch
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torchvision import transforms

from src.colors import bcolors
from src.pickle_loader import save_object

c = bcolors()


class EuroSatMS(Dataset):
    def __init__(self,
                 dataframe,
                 root_dir,
                 encoder,
                 num_aug,
                 transform,
                 augment=None,
                 select_chan=None,
                 n_jobs=-4):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.select_chan = select_chan
        self.num_aug = num_aug

        self.enc = encoder

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"\n{c.OKCYAN}Images:         {len(dataframe)}{c.ENDC}")
        print(f"{c.OKCYAN}Augmentations:  {len(dataframe) * self.num_aug}{c.ENDC}")
        print(f"{c.OKCYAN}Jobs:           {n_jobs} {c.ENDC}\n")

        start_time = time.time()
        result = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(dataframe))
        )

        self.samples = []
        self.targets = []
        self.groups = []
        self.aug_idx = []

        for x, y, idx in result:
            self.samples.append(x)
            if self.augment:
                for i in range(idx * self.num_aug, (idx + 1) * self.num_aug):
                    self.aug_idx.append(i)
                    self.groups.append(idx)
                    self.targets.append(y)
            else:
                self.targets.append(y)
                self.groups.append(idx)

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)
        end_time = time.time()
        t = end_time - start_time
        print(f"\n{c.OKBLUE}Time taken:      {int((t - (t % 60)) / 60)} min {t % 60} sec {c.ENDC}")

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])

        with rasterio.open(img_path) as src:
            image = np.array(src.read())

        image = image[self.select_chan].astype(np.float32)

        # image = image / 10000
        # image = image.clip(0, 1)

        # for channel in range(image.shape[0]):
        #     rgb_min, rgb_max = image[channel].min(), image[channel].max()
        #     if rgb_max - rgb_min == 0:
        #         image[channel] = image[channel] - rgb_min
        #     else:
        #         image[channel] = (image[channel] - rgb_min) / (rgb_max - rgb_min)
        # image = image.clip(0, 1)

        rgb_min, rgb_max = image.min(), image.max()
        image = (image - rgb_min) / (rgb_max - rgb_min)
        image = image.clip(0, 1)

        if 9 in self.select_chan:
            b10_channel = np.zeros((1, 64, 64))
            image[self.select_chan.index(9)] = b10_channel

        target = self.dataframe.iloc[idx, 1]

        return image, target, idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.samples[idx // self.num_aug]
        image = torch.tensor(image, dtype=torch.float32)

        target = self.targets[idx]
        target = self.enc.transform(np.array([target]).reshape(-1, 1)).toarray()
        target = torch.tensor(target.flatten(), dtype=torch.float32)

        if self.augment:
            image = self.augment(image)

        if self.transform:
            image = self.transform(image)

        image = image.squeeze(0)
        return image, target
