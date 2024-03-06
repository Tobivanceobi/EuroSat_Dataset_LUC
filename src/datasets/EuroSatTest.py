import sys

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
    def __init__(self, root_dir, select_chan=[3, 2, 1], transform=None, n_jobs=-4):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform
        self.select_chan = select_chan

        self.enc = load_object("data/on_hot_encoder")

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"{c.OKCYAN}Number of images: {len(self.files)}{c.ENDC}")
        print(f"{c.OKCYAN}Number of jobs:   {n_jobs} {c.ENDC}\n")

        self.samples = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(self.files))
        )

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])

        sample_id = int(self.files[idx].split(".")[0].split("_")[1])

        raw = np.load(img_path)
        image = raw[:, :, self.select_chan]

        rgb_min = image.min()
        rgb_max = image.max()

        image = (image - rgb_min) / (rgb_max - rgb_min)

        if self.transform:
            image = Image.fromarray(image.astype('uint8'))
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image, sample_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]

        return sample
