import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.nn import init


logger = logging.getLogger(__name__)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/blob/34a8c9678406a1c7dd0fec4c9f0d25d017be55fb/main.py#L325
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def create_model(model_name: str, num_classes: int):
    # Architecture
    model = ptcv_get_model(model_name, pretrained=False)
    in_features = model.output.in_features
    model.output = nn.Linear(in_features=in_features, out_features=num_classes)
    model.apply(weight_init)
    return model


class LitClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.save_hyperparameters()
        super().__init__()

        self.model = model

        # Loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._loss_helper(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._loss_helper(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._loss_helper(batch, "test")

    def _loss_helper(self, batch, phase: str = "train"):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y)

        self.log(f"loss/{phase}", loss, on_epoch=True, on_step=False)
        self.log(f"acc/{phase}", acc, on_epoch=True, on_step=False)
        return {"loss": loss, "acc": acc}

    # def _on_epoch_end_helper(self):
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.milestones
        )

        return [optimizer], [lr_scheduler]
