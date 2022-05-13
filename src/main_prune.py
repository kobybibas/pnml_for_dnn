import logging
import os
import os.path as osp
import time
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import WandbLogger
from dataset_utils import get_dataloadrs
from lit_utils import LitClassifier, create_model

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name="prune")
def main_prune(cfg: DictConfig):
    t0 = time.time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(123)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type="prune",
        name=name,
    )
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    # Datasets
    trainloader, valloader, testloader, _, classes = get_dataloadrs(
        cfg.dataset.data_dir,
        cfg.train.batch_size,
        cfg.train.num_workers,
        cfg.dataset.train_val_ratio,
    )
    for loader in [trainloader, valloader, testloader]:
        logger.info(f"{len(loader.dataset)=}")
    logger.info(f"{classes=}")

    # Model pruning
    model_init = create_model(cfg.dataset.model, len(classes))
    model_init_path = osp.join(out_dir, "model_init.ckpt")
    torch.save(model_init, model_init_path)

    # Lightning training
    erm_model = LitClassifier(model=model_init, cfg=cfg.train)
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir, monitor="acc/val", mode="max", verbose=True
    )

    pruning_callback = ModelPruning(
        "l1_unstructured",
        amount=lambda epoch_: cfg.prune_amount if epoch_ == 0 else None,
        verbose=1,
        use_lottery_ticket_hypothesis=False,
        use_global_unstructured=True,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        min_epochs=cfg.train.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=WandbLogger(experimnet=wandb.run),
        strategy="ddp",
        callbacks=[
            checkpoint_callback,
            pruning_callback,
            pl.callbacks.LearningRateMonitor(),
        ],
    )

    trainer.fit(erm_model, trainloader, valloader)
    logger.info(f"Finish training in {time.time()-t0:.2f} sec")

    logger.info(f"Test ckpt_path={checkpoint_callback.best_model_path}")
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=testloader)

    # Upload best model
    wandb.save(model_init_path)
    wandb.save(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main_prune()
