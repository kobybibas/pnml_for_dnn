import logging
import os
import os.path as osp
import time

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np
import wandb

logger = logging.getLogger(__name__)


train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def get_dataloadrs(cfg):
    trainset = CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=train_transform
    )

    testset = CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=test_transform
    )

    trainset.targets = np.array(trainset.targets)
    len_pre = len(trainset)
    mask = np.logical_or(trainset.targets == 0, trainset.targets == 1)
    trainset.data = trainset.data[mask]
    trainset.targets = trainset.targets[mask]
    len_post = len(trainset)
    logger.info(f"Reducing training set [pre post]=[{len_pre} {len_post}]")

    testset.targets = np.array(testset.targets)
    len_pre = len(testset)
    mask = np.logical_or(testset.targets == 0, testset.targets == 1)
    testset.data = testset.data[mask]
    testset.targets = testset.targets[mask]
    len_post = len(testset)
    logger.info(f"Reducing testing set [pre post]=[{len_pre} {len_post}]")

    train_loader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )

    test_loader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    classes = np.array(
        [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    )

    classes = classes[[0, 1]]

    return train_loader, test_loader, classes

