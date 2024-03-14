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

from src.colors import bcolors
from src.pickle_loader import load_object

c = bcolors()


class EuroSatTestSet(Dataset):
    def __init__(self, root_dir, select_chan, transform=None, augment=None, add_b10=True, n_jobs=-4):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform
        self.augment = augment
        self.select_chan = select_chan
        self.add_b10 = add_b10

        self.enc = load_object("data/on_hot_encoder")

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

        # if self.transform:
        #     image = torch.tensor(image, dtype=torch.float32)
        #     image = self.transform(image).squeeze(0).numpy()

        # image = image / 10000
        # image = image.clip(0, 1)

        for channel in range(image.shape[0]):
            rgb_min, rgb_max = image[channel].min(), image[channel].max()
            if rgb_max - rgb_min == 0:
                image[channel] = image[channel] - rgb_min
            else:
                image[channel] = (image[channel] - rgb_min) / (rgb_max - rgb_min)
        image = image.clip(0, 1)
        # rgb_min, rgb_max = image.min(), image.max()
        # image = (image - rgb_min) / (rgb_max - rgb_min)

        if self.add_b10:
            b10_channel = np.zeros((1, 64, 64))
            image = np.insert(image, 3, b10_channel, axis=0)

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
            image = self.augment(image).squeeze(0)

        if self.transform:
            image = self.transform(image)

        return image, samp_id
