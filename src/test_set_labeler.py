from matplotlib import pyplot as plt

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import Config

config = Config()
# Define options for user selection
options = {
    1: 'AnnualCrop',
    2: 'Forest',
    3: 'HerbaceousVegetation',
    4: 'Highway',
    5: 'Industrial',
    6: 'Pasture',
    7: 'PermanentCrop',
    8: 'Residential',
    9: 'River',
    10: 'SeaLake'
}


# Function to load and preprocess image
def load_and_preprocess_image(im_path):
    img = np.load(im_path).transpose(2, 0, 1)  # Swap axes for RGB format
    img = img[[3, 2, 1]]  # Reorder to BGR (assuming model expects BGR)
    img = img.astype(np.float32)
    rgb_min, rgb_max = img.min(), img.max()
    img = (img - rgb_min) / (rgb_max - rgb_min)
    img = img.clip(0, 1)
    plt.imshow(img.transpose(1, 2, 0))
    plt.show()
    return img

# Function to get user input with validation
def get_valid_user_input(prompt, valid_options):
    while True:
        user_input = input(prompt)
        if user_input.lower() == 't':
            return None  # Exit signal
        if user_input.lower() == 's':
            return "s"
        try:
            selection = int(user_input)
            if selection in valid_options:
                return selection
            else:
                print("Invalid option. Please choose a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Initialize empty DataFrame
if os.path.exists("labels.csv"):
    data = pd.read_csv("labels.csv")
else:
    data = pd.DataFrame(columns=['file_path', 'label'])

files = os.listdir(config.DATA_DIR + "test/NoLabel/")
# remove all files in data that are already labeled
files = [f for f in files if f not in data['file_path'].tolist()]

# Iterate over files in the folder
for filename in files:
    if filename.endswith(".npy"):  # Check for specific file extension
        file_path = os.path.join(config.DATA_DIR + "test/NoLabel/", filename)
        img = load_and_preprocess_image(file_path)

        # Print options for user selection
        print("Select a label for", filename)
        for num, label in options.items():
            print(f"{num}. {label}")

        # Get user input and validate
        selection = get_valid_user_input("Enter your choice (1-10) or 't' to stop: ", options.keys())

        # Exit if user chooses 't'
        if selection is None:
            break
        elif selection == 's':
            continue

        # Add label and file path to DataFrame
        label = options[selection]
        data = pd.concat([data, pd.DataFrame({'file_path': [filename], 'label': [label]})], ignore_index=True)

# Save DataFrame
if not data.empty:
    data.to_csv("labels.csv", index=False)
    print("Labels saved to labels.csv")
else:
    print("No labels were selected.")
