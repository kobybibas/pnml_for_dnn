import logging
import os
import time

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset_utils import get_dataloadrs
from lit_utils import LitClassifier

logger = logging.getLogger(__name__)

# TODO: 1. Train erm. Try to maximze
# TODO: 2. Efficient way to add test sample to training set. Make dataset=trainet+testset and use Subset? 
# TODO: 3. Log results: for each test sample: prob of each label, true label, regret, comparison to erm

@hydra.main(config_path="../configs/", config_name="main_train")
def main_train(cfg: DictConfig):
    t0 = time.time()
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())

    wandb.init(project="pnml_for_dnn", dir=out_dir, config=OmegaConf.to_container(cfg))
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    # Datasets
    trainloader, testloader, classes = get_dataloadrs(cfg)
    logger.info(f"{len(trainloader.dataset)=} {len(testloader.dataset)=} {classes=}")

    # Model
    model = LitClassifier(cfg)

    # Train
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        gpus=None,
        logger=WandbLogger(experimnet=wandb.run),
        strategy="ddp",
        callbacks=[ModelCheckpoint(dirpath=out_dir)],
    )
    trainer.fit(model, trainloader, testloader)
    logger.info(f"Finish training in {time.time()-t0:.2f} sec")


if __name__ == "__main__":
    main_train()
