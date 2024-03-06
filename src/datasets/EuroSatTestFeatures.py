import os

import numpy as np
from joblib import Parallel, delayed

from src.colors import bcolors
from src.pickle_loader import load_object

c = bcolors()


class EuroSatTestSetFeatures:
    def __init__(self, root_dir, n_jobs=-4):
        self.files = os.listdir(root_dir)
        self.root_dir = root_dir

        self.enc = load_object("data/label_encoder")

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
        image = raw[:, :, [3, 2, 1]]

        rgb_min = image.min()
        rgb_max = image.max()

        image = (image - rgb_min) / (rgb_max - rgb_min)

        hist_red = np.histogram(image[:, :, 0], bins=100)[0]
        hist_green = np.histogram(image[:, :, 1], bins=100)[0]
        hist_blue = np.histogram(image[:, :, 2], bins=100)[0]
        features = np.concatenate([hist_red, hist_green, hist_blue])

        return features, sample_id
