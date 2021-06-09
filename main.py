# +
# machine learning
import torch
import torch.nn as nn
from torchvision import transforms

# torch-API
import pytorch_lightning as pl

# logging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

# callbacks
from pytorch_lightning.callbacks import EarlyStopping

# configuration
from omegaconf import OmegaConf

# visualization
import matplotlib.pyplot as plt

# NNs and pytorch-lightning-module
from module import ImageModule
import timm

# pytorch-lightning-data-module
from dataset import ImageDataModule

# augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
# -

transform = A.Compose([
    #A.Resize(32,32),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=10, p=0.5),
    A.Normalize(
        mean = [0.5,0.5,0.5],
        std = [0.5, 0.5, 0.5]
        ), 
    ])

target_transform = None #transforms.Compose([])

# +
cfg = dict(
    project_name = 'Cifar10',
    name = 'efficientnet_lite0_test1',
    model = 'efficientnet_lite0',
    epochs = 64,
    batch_size = 64,
    lr=0.002,
    
    dataset='CIFAR10',
    num_classes=10,
    train_split=0.9,
    
    weight_decay = 1e-2,
    auto_lr_find=False,
)

cfg = OmegaConf.create(cfg)

# +
dataset = ImageDataModule(cfg, transform=transform, target_transform=target_transform)

net = timm.create_model(cfg.model, pretrained=True, num_classes=cfg.num_classes, in_chans=3)
model = ImageModule(net, cfg)

# +
# logger
wandb_logger = WandbLogger(name='CNN_big',
                           project='CIFAR10', 
                           offline=False, 
                           log_model=True)


early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.00, 
                           patience=8, 
                           verbose=True,
                           mode='min')

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(gpus=1, 
                     max_epochs=cfg.epochs, 
                     progress_bar_refresh_rate=10,
                     callbacks=[lr_monitor, early_stop],
                     logger=wandb_logger,
                     auto_lr_find=cfg.auto_lr_find)

# +
if cfg.auto_lr_find:
    trainer.tune(model, datamodule=dataset)
    
trainer.fit(model, dataset)
# -


