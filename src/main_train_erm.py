import logging
import os
import os.path as osp
import time

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset_utils import get_dataloadrs
from lit_utils import LitClassifier, create_model

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name="train_erm")
def main_train_erm(cfg: DictConfig):
    t0 = time.time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(123)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type="train_erm",
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

    # Model
    model = create_model(cfg.dataset.model, len(classes))
    erm_model = LitClassifier(model=model, cfg=cfg.train)

    model_init_path = osp.join(out_dir, "model_init.ckpt")
    torch.save(model, model_init_path)

    model_init_state_dict_path = osp.join(out_dir, "model_init_state_dict.ckpt")
    torch.save(model.state_dict(), model_init_state_dict_path)

    # ERM Training
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir, monitor="acc/val", mode="max", verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        min_epochs=cfg.train.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=WandbLogger(experimnet=wandb.run),
        strategy="ddp",
        callbacks=[checkpoint_callback, pl.callbacks.LearningRateMonitor()],
    )

    # Train
    trainer.fit(erm_model, trainloader, valloader)
    logger.info(f"Finish training in {time.time()-t0:.2f} sec")

    # Test
    logger.info(f"Test ckpt_path={checkpoint_callback.best_model_path}")
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=testloader)

    # Upload best model
    wandb.save(model_init_path)
    wandb.save(model_init_state_dict_path)
    wandb.save(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main_train_erm()
