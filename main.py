import sys

import pandas as pd
import torch
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split

from config import Config
from src.colors import bcolors
from torchvision import transforms

from src.datasets.EuroSatMS import EuroSatMS
from src.datasets.EuroSatMSFeatures import EuroSatMSFeatures
from src.datasets.EuroSatTest import EuroSatTestSet
from src.training.data import EuroSatDataModule
from src.training.pretrainedModels import EuroSatPreTrainedModel
import torchvision.transforms as T
import torch.nn as nn

c = bcolors()

config = Config()
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 0.0001

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentation = transforms = nn.Sequential(
    T.RandomVerticalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=0, translate=None, shear=0, scale=(1, 1.2)),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(config.TRAIN_FILE)
df_eval = pd.read_csv(config.TEST_FILE)
dataset = EuroSatTestSet(config.TEST_MS_DIR)

print(dataset[0][0].shape)

# dataset = EuroSatMS(df[df['label'] == 'Industrial'].head(20), config.TRAIN_MS_DIR)
sys.exit()

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Initialize the LightningModule and DataModule
print(f"""{c.OKGREEN}Initializing the model...{c.ENDC}""")
lightning_model = EuroSatResNet(num_classes=10, learning_rate=LEARNING_RATE)

print(f"""{c.OKGREEN}Initializing the data module...{c.ENDC}""")
data_module = EuroSatDataModule(train_df, val_df, config.TRAIN_MS_DIR, BATCH_SIZE)

# Initialize the Trainer
trainer = Trainer(max_epochs=N_EPOCHS, accelerator="gpu", devices=1)

# Train the model
trainer.fit(lightning_model, datamodule=data_module)

# Test the model
trainer.validate(lightning_model, datamodule=data_module)



