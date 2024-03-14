import os

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog

from src.colors import bcolors
from src.pickle_loader import load_object

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


class EuroSatTestSetFeatures:
    def __init__(self, root_dir, channels, n_jobs=-4):
        self.files = os.listdir(root_dir)
        self.root_dir = root_dir
        self.channels = channels

        self.enc = load_object("data/label_encoder")

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"{c.OKCYAN}Number of images: {len(self.files)}{c.ENDC}")
        print(f"{c.OKCYAN}Number of jobs:   {n_jobs} {c.ENDC}\n")

        # Process images in parallel
        samples = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(self.files))
        )

        self.x_data = []
        self.y_data = []

        for x, y in samples:
            self.x_data.append(x)
            self.y_data.append(y)

        self.x_data = np.array(self.x_data).astype(np.float32)
        self.y_data = np.array(self.y_data).astype(np.float32)

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])

        sample_id = int(self.files[idx].split(".")[0].split("_")[1])

        image = np.load(img_path).transpose(2, 0, 1)
        image = image[self.channels].astype(np.float32)

        rgb_min, rgb_max = image.min(), image.max()
        image = (image - rgb_min) / (rgb_max - rgb_min)

        features = extract_hog_features(image)

        return features, sample_id
