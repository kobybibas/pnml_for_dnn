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

from dataset_utils import get_dataloadrs
from lit_utils import LitClassifier, create_model, predict_single_img

logger = logging.getLogger(__name__)


def execute_train_model(cfg, prune_amount: float, model_init, train_datalodaer):
    lit_model = LitClassifier(copy.deepcopy(model_init), cfg)

    pruning_callback = ModelPruning(
        "l1_unstructured",
        amount=lambda epoch_: prune_amount if epoch_ == 0 else None,
        verbose=1,
        use_lottery_ticket_hypothesis=False,
        use_global_unstructured=True,
    )

    # ERM Training
    genie_trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[pruning_callback] if prune_amount > 0.0 else None,
        strategy="ddp",
        # For speed up: https://william-falcon.medium.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        replace_sampler_ddp=False,
        precision=16,
    )
    genie_trainer.fit(lit_model, train_datalodaer)
    lit_model.eval()
    return lit_model


def get_genie_probs(
    cfg, prune_amount, model_init, pnml_trainloader, num_classes: int
) -> torch.Tensor:
    test_img, _ = pnml_trainloader.dataset.get_test_sample()

    genie_probs = []
    for class_idx in range(num_classes):
        pnml_trainloader.dataset.set_pseudo_test_label(class_idx)
        genie_model = execute_train_model(
            cfg, prune_amount, model_init, pnml_trainloader
        )

        # Predict
        probs = predict_single_img(genie_model, test_img)
        genie_probs.append(probs[class_idx].item())

    return torch.tensor(genie_probs)


@hydra.main(config_path="../configs/", config_name="main_pnml")
def main_pnml(cfg: DictConfig):
    t0 = time.time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(123)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type="pnml",
        name=name,
    )
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    # Datasets
    trainloader, valloader, testloader, pnml_train_loader, classes = get_dataloadrs(
        cfg.dataset.data_dir,
        cfg.train.batch_size,
        cfg.train.num_workers,
        cfg.dataset.train_val_ratio,
    )
    for loader in [trainloader, valloader, testloader, pnml_train_loader]:
        logger.info(f"{len(loader.dataset)=}")
    logger.info(f"{classes=}")

    # Initalize model
    model_init = create_model(cfg.dataset.model, len(classes))
    model_init_path = osp.join(out_dir, "model_init.ckpt")
    torch.save(model_init, model_init_path)

    # ERM training
    erm_model = execute_train_model(
        cfg.train, cfg.prune_amount, model_init, trainloader
    )

    # Create logging table
    result_tracker_h = ResultTracker(out_dir, wandb)
    res_df = pd.DataFrame(
        columns=[
            "test_idx",
            "pnml_nf",
            "pnml_regret",
            "pnml_logloss",
            "pnml_is_correct",
            "pnml_probs",
            "genie_probs",
            "erm_logloss",
            "erm_is_correct",
            "erm_probs",
        ]
    )

    # pNML training:
    num_classes = len(classes)
    for test_idx in range(pnml_train_loader.dataset.num_test_samples):
        t1 = time.time()
        pnml_train_loader.dataset.set_test_idx(test_idx)
        test_img, _ = pnml_train_loader.dataset.get_test_sample()
        test_true_label = pnml_train_loader.dataset.get_true_test_label()
        erm_probs = predict_single_img(erm_model, test_img)

        genie_probs = get_genie_probs(
            cfg.train, cfg.prune_amount, model_init, pnml_train_loader, num_classes
        )
        result_tracker_h.calc_test_sample_performance(
            test_idx, test_true_label, genie_probs, erm_probs
        )
        result_tracker_h.finish_sample()
        logger.info(
            f"[{test_idx}/{pnml_train_loader.dataset.num_test_samples -1}] Finish in {time.time()-t1:.2f} sec"
        )


if __name__ == "__main__":
    main_pnml()
