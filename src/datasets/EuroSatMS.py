import os

import numpy as np
import rasterio
import torch
from PIL import Image
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torchvision import transforms

from src.colors import bcolors
from src.pickle_loader import save_object

c = bcolors()


class EuroSatMS(Dataset):
    def __init__(self, dataframe, root_dir, augment=None, transform=None, select_chan=None, n_jobs=-4):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.select_chan = select_chan

        self.enc = OneHotEncoder()
        self.enc = self.enc.fit(dataframe[['label']].values.reshape(-1, 1))
        save_object(self.enc, "data/on_hot_encoder")

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"{c.OKCYAN}Number of images: {len(dataframe)}{c.ENDC}")
        print(f"{c.OKCYAN}Number of jobs:   {n_jobs} {c.ENDC}\n")

        # Process images in parallel
        self.samples = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(dataframe))
        )

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])

        with rasterio.open(img_path) as src:
            image = np.array(src.read()).transpose(1, 2, 0)

        image = image[:, :, self.select_chan]
        rgb_min = image.min()
        rgb_max = image.max()

        image = (image - rgb_min) / (rgb_max - rgb_min)

        label = self.dataframe.iloc[idx, 1]
        target = self.enc.transform(np.array([label]).reshape(-1, 1)).toarray()

        return image, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, target = self.samples[idx]

        if self.transform:
            image = Image.fromarray(image.astype('uint8'))
            image = self.augment(image)
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        target = torch.tensor(target.flatten(), dtype=torch.float32)

        return image, target
