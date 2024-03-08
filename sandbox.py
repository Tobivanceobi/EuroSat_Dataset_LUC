import os
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed

from config import Config
from src.pickle_loader import save_object


config = Config()
df = pd.read_csv(config.TRAIN_FILE)
select_chan = [i for i in range(0, 13)]


def process_image(idx):
    if idx % 1000 == 0:
        print(f"Processing image {idx}...")
    img_path = os.path.join(config.TRAIN_MS_DIR, df.iloc[idx, 0])

    with rasterio.open(img_path) as src:
        image = np.array(src.read()).transpose(1, 2, 0)

    image = image[:, :, select_chan]
    rgb_min = image.min()
    rgb_max = image.max()

    image = (image - rgb_min) / (rgb_max - rgb_min)

    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))

    return mean, std


channel_stats = Parallel(n_jobs=-4)(
    delayed(process_image)(idx) for idx in range(len(df))
)

means = np.array([stat[0] for stat in channel_stats])
stds = np.array([stat[1] for stat in channel_stats])

overall_mean = np.mean(means, axis=0)
overall_std = np.mean(stds, axis=0)

print("Channel Index:   ", select_chan)
print("Overall Mean:    ", list([round(i, 3) for i in overall_mean]))
print("Overall Std Dev: ", list([round(i, 3) for i in overall_std]))

save_object(dict(mean=overall_mean, std=overall_std, idx=select_chan), "data/eurosat_ms_mean_std")

