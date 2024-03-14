import sys

import numpy as np
import pandas as pd
import rasterio
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage import data, exposure
from joblib import Parallel, delayed
from skimage.feature import hog

from src.colors import bcolors
from src.pickle_loader import save_object

c = bcolors()


def extract_histogram_features(image):
    hist_red = np.histogram(image[:, :, 0], bins=100)[0]
    hist_green = np.histogram(image[:, :, 1], bins=100)[0]
    hist_blue = np.histogram(image[:, :, 2], bins=100)[0]
    features = np.concatenate([hist_red, hist_green, hist_blue])
    return features


def extract_hog_features(image):
    features = hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
        channel_axis=0
    )
    return features


class EuroSatMSFeatures:
    def __init__(self, dataframe, channels, root_dir, encoder, n_jobs=-4):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.channels = channels

        self.enc = encoder

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"{c.OKCYAN}Number of images: {len(dataframe)}{c.ENDC}")
        print(f"{c.OKCYAN}Number of jobs:   {n_jobs} {c.ENDC}\n")

        # Process images in parallel
        samples = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(dataframe))
        )

        self.x_data = []
        self.y_data = []

        for x, y in samples:
            self.x_data.append(x)
            self.y_data.append(y)

        self.x_data = np.array(self.x_data).astype(np.float32)
        self.y_data = np.array(self.y_data).astype(np.float32)

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])

        with rasterio.open(img_path) as src:
            image = np.array(src.read())

        image = image[self.channels].astype(np.float32)

        rgb_min, rgb_max = image.min(), image.max()
        image = (image - rgb_min) / (rgb_max - rgb_min)

        features = extract_hog_features(image)

        label = self.dataframe.iloc[idx, 1]
        target = self.enc.transform([label])[0]

        return features, target
