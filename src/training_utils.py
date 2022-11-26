import copy
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import WandbLogger

from lit_utils import LitClassifier, predict_single_img

logger = logging.getLogger(__name__)


def execute_train_model(
    cfg,
    prune_amount: float,
    model_init,
    train_dataloader,
    val_dataloader=None,
    out_dir: str = None,
    is_print: bool = False,
):
    lit_model = LitClassifier(copy.deepcopy(model_init), cfg)

    pruning_callback = ModelPruning(
        "l1_unstructured",
        amount=lambda epoch_: prune_amount if epoch_ == 0 else None,
        verbose=1,
        use_lottery_ticket_hypothesis=False,
        use_global_unstructured=True,
        make_pruning_permanent=False,
    )

    callbacks = []
    enable_checkpointing = False
    if out_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=out_dir,
            filename="model_epoch_{epoch}_val_loss_{loss/val:.2f}",
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        )
        callbacks += [checkpoint_callback]
        enable_checkpointing = True
    if prune_amount > 0.0:
        callbacks += [pruning_callback]

    # ERM Training
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        strategy="ddp",
        enable_progress_bar=is_print,
        enable_model_summary=is_print,
        enable_checkpointing=enable_checkpointing,
        logger=WandbLogger() if is_print else False,
        replace_sampler_ddp=False,
        precision=16,
    )
    trainer.fit(lit_model, train_dataloader, val_dataloader)

    # Take best preforming model on validation set
    if out_dir is not None:
        logger.info(f"load_from_checkpoint: {checkpoint_callback.best_model_path}")
        lit_model.load_from_checkpoint(checkpoint_callback.best_model_path)

    # Inference
    lit_model.eval()
    return lit_model


def get_genie_probs(
    cfg,
    prune_amount,
    model_init,
    pnml_trainloader,
    num_classes: int,
    is_print: bool = False,
) -> torch.Tensor:
    test_img, _ = pnml_trainloader.dataset.get_test_sample()

    genie_probs = []
    for class_idx in range(num_classes):
        pnml_trainloader.dataset.set_pseudo_test_label(class_idx)
        genie_model = execute_train_model(
            cfg, prune_amount, model_init, pnml_trainloader, is_print=is_print
        )

        # Predict
        probs = predict_single_img(genie_model, test_img)
        genie_probs.append(probs[class_idx].item())

    return torch.tensor(genie_probs)
