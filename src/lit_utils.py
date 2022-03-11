import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch


logger = logging.getLogger(__name__)


class LitClassifier(pl.LightningModule):
    def __init__(self, cfg, num_classes: int = 10):
        self.cfg = cfg
        self.num_classes = num_classes
        self.save_hyperparameters()
        super().__init__()

        # Architecture
        self.model = ptcv_get_model(self.cfg.model, pretrained=False)
        in_features = self.model.output.in_features
        self.model.output = nn.Linear(in_features=in_features, out_features=num_classes)

        # Loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._loss_helper(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._loss_helper(batch, "val")

    def _loss_helper(self, batch, phase: str = "train"):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y)

        self.log(f"loss/{phase}", loss, on_epoch=True, on_step=False)
        self.log(f"acc/{phase}", acc, on_epoch=True, on_step=False)
        return {"loss": loss, "acc": acc}

    def _on_epoch_end_helper(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.milestones
        )

        return [optimizer], [lr_scheduler]
