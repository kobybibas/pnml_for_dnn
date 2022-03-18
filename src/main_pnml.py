import logging
import os
import time
import pandas as pd
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset_utils import get_dataloadrs
from lit_utils import LitClassifier

logger = logging.getLogger(__name__)


def calc_performance(probs, true_label, learner_name: str):
    is_correct = torch.argmax(probs) == true_label
    logloss = -torch.log(probs[true_label])
    return {
        learner_name + "is_correct": is_correct.item(),
        learner_name + "logloss": logloss.item(),
    }


@hydra.main(config_path="../configs/", config_name="main_pnml")
def main_pnml(cfg: DictConfig):
    t0 = time.time()
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())

    wandb.init(project="pnml_for_dnn", dir=out_dir, config=OmegaConf.to_container(cfg))
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    # Datasets
    trainloader, testloader, pnml_train_loader, classes = get_dataloadrs(cfg)
    for loader in [trainloader, testloader, pnml_train_loader]:
        logger.info(f"{len(loader.dataset)=}")
    logger.info(f"{classes=}")

    # Model
    # TODO load preteined model
    erm_model = LitClassifier(cfg, len(classes))

    # Create logging table
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
    pl.utilities.distributed.log.setLevel(logging.ERROR)
    num_classes = len(classes)
    for test_idx in range(pnml_train_loader.dataset.num_test_samples):
        test_img, _ = pnml_train_loader.dataset.get_test_sample()
        test_true_label = pnml_train_loader.dataset.get_true_test_label()
        with torch.no_grad():
            erm_probs = torch.nn.functional.softmax(
                erm_model(test_img.unsqueeze(0)), dim=-1
            ).squeeze()

        genie_probs = []
        for class_idx in range(num_classes):
            pnml_train_loader.dataset.set_test_idx(test_idx)
            pnml_train_loader.dataset.set_pseudo_test_label(class_idx)

            genie_model = LitClassifier(cfg, len(classes))

            # ERM Training
            genie_trainer = pl.Trainer(
                max_epochs=cfg.epochs,
                min_epochs=cfg.epochs,
                gpus=None,
                strategy="ddp",
                enable_progress_bar=True,
                enable_model_summary=False,
            )
            genie_trainer.fit(genie_model, trainloader)
            genie_model.eval()

            # Predict
            with torch.no_grad():
                probs = torch.nn.functional.softmax(
                    genie_model(test_img.unsqueeze(0)), dim=-1
                ).squeeze()
            genie_probs.append(probs[class_idx].item())
        genie_probs = torch.tensor(genie_probs)
        pnml_nf = torch.sum(genie_probs)
        pnml_regret = torch.log(pnml_nf)
        pnml_probs = genie_probs / pnml_nf

        # pNML
        pnml_dict = calc_performance(pnml_probs, test_true_label, learner_name="pnml_")
        pnml_dict["pnml_nf"] = pnml_nf.item()
        pnml_dict["pnml_regret"] = pnml_regret.item()
        pnml_dict["pnml_probs"] = pnml_probs.tolist()
        pnml_dict["genie_probs"] = genie_probs.tolist()

        # ERM
        erm_dict = calc_performance(erm_probs, test_true_label, learner_name="erm_")
        erm_dict["erm_probs"] = erm_probs.tolist()

        # Combine
        res_dict = {**pnml_dict, **erm_dict}
        res_dict["test_idx"] = test_idx
        res_dict["test_true_label"] = test_true_label
        res_df = res_df.append(res_dict, ignore_index=True)

        # Upload
        logger.info(res_dict)
        test_table = wandb.Table(dataframe=res_df)
        wandb.run.log({"test_table": test_table})


if __name__ == "__main__":
    main_pnml()
