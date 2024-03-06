import sys

import numpy as np
import pandas as pd
import rasterio
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from joblib import Parallel, delayed

from src.colors import bcolors
from src.pickle_loader import save_object

c = bcolors()


class EuroSatMSFeatures:
    def __init__(self, dataframe, root_dir, n_jobs=-4):
        self.dataframe = dataframe
        self.root_dir = root_dir

        self.enc = LabelEncoder()
        self.enc = self.enc.fit(dataframe[['label']].values.flatten())
        save_object(self.enc, "data/label_encoder")

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"{c.OKCYAN}Number of images: {len(dataframe)}{c.ENDC}")
        print(f"{c.OKCYAN}Number of jobs:   {n_jobs} {c.ENDC}\n")

        # Process images in parallel
        self.samples = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(dataframe))
        )

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])

        # Read the .tif file
        with rasterio.open(img_path) as src:
            # Select the RGB bands of the multi spectral image
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)

            image = np.dstack((red, green, blue))

        rgb_min = image.min()
        rgb_max = image.max()

        image = (image - rgb_min) / (rgb_max - rgb_min)

        hist_red = np.histogram(image[:, :, 0], bins=100)[0]
        hist_green = np.histogram(image[:, :, 1], bins=100)[0]
        hist_blue = np.histogram(image[:, :, 2], bins=100)[0]
        features = np.concatenate([hist_red, hist_green, hist_blue])

        label = self.dataframe.iloc[idx, 1]
        target = self.enc.transform([label])[0]

        return features, target
