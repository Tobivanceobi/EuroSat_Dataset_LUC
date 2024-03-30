import sys
import time

import numpy as np
import pandas as pd
import rasterio
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
from torchvision.transforms import transforms

from config import Config
from src.colors import bcolors
from src.pickle_loader import load_object

c = bcolors()
config = Config()


def normalize(img, means, stds):
    if len(means) == 3:
        for i, (mean, std) in enumerate(zip(means, stds)):
            min_value = mean - 2 * std
            max_value = mean + 2 * std
            img[i] = (img[i] - min_value) / (max_value - min_value) * 255.0
            img[i] = np.clip(img[i], 0, 255).astype(np.uint8)
    return img


class EuroSatTestSet(Dataset):
    def __init__(self, root_dir, select_chan, transform=None, mean_std=None, augment=None, add_B10=True, n_jobs=-4):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform
        self.augment = augment
        self.select_chan = select_chan
        self.add_B10 = add_B10

        self.mean_std = mean_std

        self.enc = load_object(config.DATA_DIR + "on_hot_encoder")

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"{c.OKCYAN}Number of images: {len(self.files)}{c.ENDC}")
        print(f"{c.OKCYAN}Number of jobs:   {n_jobs} {c.ENDC}\n")

        start_time = time.time()
        result = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(self.files))
        )

        self.samples = []
        self.sample_idx = []

        for x, idx in result:
            self.samples.append(x)
            self.sample_idx.append(idx)

        self.samples = np.array(self.samples).astype(np.float32)
        end_time = time.time()
        t = end_time - start_time
        print(f"\n{c.OKBLUE}Time taken:      {int((t - (t % 60)) / 60)} min {t % 60} sec {c.ENDC}")

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])

        sample_id = int(self.files[idx].split(".")[0].split("_")[1])

        image = np.load(img_path).transpose(2, 0, 1)
        image = image[self.select_chan].astype(np.float32)

        # if self.mean_std:
        #     image = normalize(image, self.mean_std['mean'], self.mean_std['std'])
        #
        # rgb_min, rgb_max = image.min(), image.max()
        # image = (image - rgb_min) / (rgb_max - rgb_min)
        # image = image / 10000
        # image = image.clip(0, 1)

        for channel in range(len(self.select_chan)):
            rgb_min, rgb_max = image[channel].min(), image[channel].max()
            image[channel] = (image[channel] - rgb_min) / (rgb_max - rgb_min)

        if self.add_B10:
            B10_channel = np.zeros((1, 64, 64))
            image = np.insert(image, 3, B10_channel, axis=0)

        return image, sample_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.samples[idx]
        image = torch.tensor(image, dtype=torch.float32)
        samp_id = self.sample_idx[idx]

        if self.augment:
            image = self.augment(image)

        if self.transform:
            image = self.transform(image)

        image = image.squeeze(0)
        return image, samp_id
