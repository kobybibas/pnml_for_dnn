import copy
import logging
import os
import os.path as osp
import time
from result_utils import ResultTracker
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.loggers import WandbLogger
from dataset_utils import get_dataloadrs
from lit_utils import LitClassifier, create_model, predict_single_img

logger = logging.getLogger(__name__)


def execute_train_model(
    cfg, prune_amount: float, model_init, train_datalodaer, is_print: bool = False
):
    lit_model = LitClassifier(copy.deepcopy(model_init), cfg)

    pruning_callback = ModelPruning(
        "l1_unstructured",
        amount=lambda epoch_: prune_amount if epoch_ == 0 else None,
        verbose=1,
        use_lottery_ticket_hypothesis=False,
        use_global_unstructured=True,
        make_pruning_permanent=False
    )

    # ERM Training
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[pruning_callback] if prune_amount > 0.0 else None,
        strategy="ddp",
        # For speed up: https://william-falcon.medium.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719
        enable_progress_bar=is_print,
        enable_model_summary=is_print,
        enable_checkpointing=False,
        logger=WandbLogger() if is_print else False,
        replace_sampler_ddp=False,
        precision=16,
    )
    trainer.fit(lit_model, train_datalodaer)
    lit_model.eval()
    return lit_model

def get_genie_probs(
    cfg, prune_amount, model_init, pnml_trainloader, num_classes: int,is_print:bool=False
) -> torch.Tensor:
    test_img, _ = pnml_trainloader.dataset.get_test_sample()

    genie_probs = []
    for class_idx in range(num_classes):
        pnml_trainloader.dataset.set_pseudo_test_label(class_idx)
        genie_model = execute_train_model(
            cfg, prune_amount, model_init, pnml_trainloader,is_print=is_print
        )

        # Predict
        probs = predict_single_img(genie_model, test_img)
        genie_probs.append(probs[class_idx].item())

    return torch.tensor(genie_probs)