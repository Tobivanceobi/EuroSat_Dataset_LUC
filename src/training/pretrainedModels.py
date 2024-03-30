import numpy as np
import pytorch_lightning as pl
import torch
from kornia.constants import Resample
from tabulate import tabulate
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ResNet50_Weights
import torchvision
from torchvision import transforms
import torchgeo.models as models
from torchgeo.models import ResNet18_Weights, ResNet50_Weights
from torch import optim, nn
from torch.optim.lr_scheduler import ExponentialLR
import kornia.augmentation as K

from src.colors import bcolors

c = bcolors()


class EuroSatPreTrainedModel(pl.LightningModule):
    def __init__(self,
                 backbone,
                 dropout,
                 layers,
                 learning_rate,
                 weight_decay,
                 opt="adam",
                 freeze_backbone=True,
                 momentum=0.9,
                 gamma=0.9,
                 n_classes=10):
        super(EuroSatPreTrainedModel, self).__init__()

        self.backbone = backbone
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gamma = gamma
        self.opt = opt

        self.criterion = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential()

        self.accuracy = 0.0
        self.ep_out = []
        self.ep_true = []
        self.num_features = 0

        # Get the number of features in the last layer of the backbone
        if self.backbone.__class__.__name__ == "AlexNet":
            self.num_features = self.backbone.classifiergit .in_features
            self.backbone.classifier[-1] = nn.Identity()
        elif self.backbone.__class__.__name__ == "VisionTransformer":
            self.num_features = self.backbone.heads[0].in_features
            self.backbone.heads = nn.Identity()
        else:
            self.num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Cunstruct the classifier
        if len(layers) > 0:
            for i, layer in enumerate(layers):
                self.classifier.add_module(f"fc_{i}", nn.Linear(self.num_features, layer))
                self.classifier.add_module(f"relu_{i}", nn.ReLU())
                self.classifier.add_module(f"dropout_{i}", nn.Dropout(p=dropout))
                self.num_features = layer
            self.classifier.add_module(f"fc_out", nn.Linear(self.num_features, n_classes))
        else:
            self.classifier = nn.Linear(self.num_features, n_classes)

        # Freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Unfreeze the classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_model_layers(self, n_layers):
        print(f"{c.OKGREEN}Unfreezing {n_layers} layers of the backbone...{c.ENDC}")
        for param in self.backbone.encoder.ln.parameters():
            param.requires_grad = True
        n_layers -= 1

        if n_layers > 0:
            total_layers = len(self.backbone.encoder.layers)
            first_layer_to_unfreeze = max(0, total_layers - n_layers)

            for i in range(first_layer_to_unfreeze, total_layers):
                layer = self.backbone.encoder.layers[i]
                for param in layer.parameters():
                    param.requires_grad = True

        tabel = []
        for name, param in self.backbone.named_parameters():
            num_params = f"{param.numel() // 1000}k"
            tabel.append([f"{c.OKGREEN}{name}{c.ENDC}", f"{c.OKBLUE}{param.requires_grad}{c.ENDC}", num_params])

        print(tabulate(tabel, headers=[f"{c.OKGREEN}Layer{c.ENDC}", f"{c.OKBLUE}Trainable{c.ENDC}", f"Parameters"]))

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        _, labels = torch.max(labels, 1)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.ep_out = []
        self.ep_true = []

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        _, labels = torch.max(labels, 1)
        _, predicted = torch.max(outputs, 1)
        self.ep_out.append(predicted.cpu().numpy())
        self.ep_true.append(labels.cpu().numpy())
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        all_preds = np.concatenate(self.ep_out)
        all_true = np.concatenate(self.ep_true)
        correct = np.sum(all_preds == all_true)
        self.accuracy = correct / len(all_true)
        self.log('val_acc', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        _, labels = torch.max(labels, 1)
        _, predicted = torch.max(outputs, 1)

        loss = self.criterion(outputs, labels)

        self.ep_out.append(predicted.cpu().numpy())
        self.ep_true.append(labels.cpu().numpy())

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.accuracy, prog_bar=True, on_step=True, on_epoch=True)

    def on_test_epoch_end(self):
        all_preds = np.concatenate(self.ep_out)
        all_true = np.concatenate(self.ep_true)
        correct = np.sum(all_preds == all_true)
        self.accuracy = correct / len(all_true)
        self.log('test_accuracy', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.ep_out = []
        self.ep_true = []

    def configure_optimizers(self):
        if self.opt == "sgd":
            optimizer = optim.SGD(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        else:
            optimizer = optim.Adam(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]


def get_pretrained_model(model_name):
    tf_0 = transforms.Compose([
        K.Resize(256, resample=Resample.BILINEAR.BILINEAR),
        K.CenterCrop(224),
        K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tf_1 = transforms.Compose([
        K.RandomHorizontalFlip(p=0.75),
        K.RandomVerticalFlip(p=0.75),
        K.RandomAffine(degrees=30, translate=None, scale=None, shear=None, resample="nearest", padding_mode=2, p=0.75),
        K.RandomShear(shear=0.2, resample="nearest", padding_mode=2, p=0.75),
        K.RandomBrightness((0.6, 1.4), p=0.6),
        K.RandomContrast(contrast=(0.6, 1.4), p=0.6),
        K.RandomSaturation((0.6, 1.4), p=0.6),
        K.RandomResizedCrop((56, 56)),
    ])

    if "resnet18_RGB_MOCO" in model_name:
        return models.resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO), tf_0
    elif "resnet50_RGB_MOCO" in model_name:
        return models.resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO), tf_1
    elif "resnet50_RGB_SECO" in model_name:
        return models.resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_SECO), tf_0
    elif "resnet50_RGB" in model_name:
        tf = transforms.Compose([
            K.Resize(256, resample=Resample.BILINEAR.BILINEAR),
            K.CenterCrop(224),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2), tf
    elif "vit_b_16" in model_name:
        tf = transforms.Compose([
            K.Resize(256, resample=Resample.BILINEAR.BILINEAR),
            K.CenterCrop(224),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1), tf
    elif "vit_b_32" in model_name:
        tf = transforms.Compose([
            K.Resize(256, resample=Resample.BILINEAR.BILINEAR),
            K.CenterCrop(224),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return torchvision.models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1), tf
    elif "vit_l_16" in model_name:
        tf = transforms.Compose([
            K.Resize(242, resample=Resample.BILINEAR.BILINEAR),
            K.CenterCrop(224),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return torchvision.models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1), tf
    elif "vit_l_32" in model_name:
        tf = transforms.Compose([
            K.Resize(256, resample=Resample.BILINEAR.BILINEAR),
            K.CenterCrop(224),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return torchvision.models.vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1), tf
