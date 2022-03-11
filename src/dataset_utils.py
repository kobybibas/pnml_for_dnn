import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


train_transform = transforms.Compose(
    [
        #    transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class DatasetPnml(torch.utils.data.Dataset):
    def __init__(self, train_dataset, test_dataset) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_test_samples = len(self.train_dataset)
        self.num_train_samples = len(self.test_dataset)

        self.test_idx = 0
        self.pseudo_test_label = 0

    def set_test_idx(self, test_idx):
        self.test_idx = test_idx

    def set_pseudo_test_label(self, pseudo_test_label):
        self.pseudo_test_label = pseudo_test_label

    def __len__(self):
        # Training set size plus a single test sample
        return len(self.train_dataset) + 1

    def get_test_sample(self):
        img, target = self.test_dataset[self.test_idx]
        target = self.pseudo_test_label
        return img, target

    def get_true_test_label(self):
        _, target = self.test_dataset[self.test_idx]
        return target

    def __getitem__(self, idx):
        if idx == len(self) - 1:  # Accessing the test sample
            img, target = self.get_test_sample()
        else:
            img, target = self.train_dataset[idx]
        return img, target


def keep_only_label_subset(dataset, labels_to_keep: list):
    dataset.targets = np.array(dataset.targets)
    len_pre = len(dataset)

    mask = np.logical_or.reduce(
        [dataset.targets == label2keep for label2keep in labels_to_keep]
    )
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    len_post = len(dataset)
    logger.info(f"Reducing set [pre post]=[{len_pre} {len_post}]")
    return dataset


def get_dataloadrs(cfg):
    labels_to_keep = [0, 1]
    train_dataset = CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=train_transform
    )

    test_dataset = CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=test_transform
    )

    train_dataset = keep_only_label_subset(train_dataset, labels_to_keep)
    test_dataset = keep_only_label_subset(test_dataset, labels_to_keep)

    pnml_dataset = DatasetPnml(train_dataset=train_dataset, test_dataset=test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    pnml_train_loader = DataLoader(
        pnml_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
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

    classes = classes[labels_to_keep]

    return train_loader, test_loader, pnml_train_loader, classes

