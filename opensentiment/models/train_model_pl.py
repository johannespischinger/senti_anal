import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from opensentiment.utils import get_project_root

logger = logging.getLogger(__name__)


def build_callbacks(cfg: omegaconf.DictConfig) -> List[pl.Callback]:
    callbacks: List[pl.Callback] = []

    callbacks.append(pl.callbacks.progress.TQDMProgressBar(refresh_rate=20))

    hydra.utils.log.info("Adding callback <ModelCheckpoint>")
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            save_top_k=cfg.train.model_checkpoints.save_top_k,
            verbose=cfg.train.model_checkpoints.verbose,
        )
    )

    return callbacks


def train(
    cfg: omegaconf.DictConfig, hydra_dir: os.path, use_val_test=True
) -> Tuple[Dict, str]:
    # based on https://github.com/lucmos/nn-template/blob/main/src/run.py
    if cfg.train.deterministic:
        pl.seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0
        cfg.data.datamodule.only_take_every_n_sample = 128

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # prepare data module
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    data_module.prepare_data()
    data_module.setup("fit")

    sizes = [
        (k, len(k()))
        for k in [
            data_module.train_dataloader,
            data_module.val_dataloader,
            data_module.test_dataloader,
        ]
    ]
    hydra.utils.log.info(f"lenght of dataloaders in batches {sizes}")

    # prepare model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        **{
            "model_name_or_path": cfg.data.datamodule.model_name_or_path,
            "train_batch_size": cfg.data.datamodule.batch_size.train,
        },
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Instantiate the callbacks
    callbacks: List[pl.Callback] = build_callbacks(cfg=cfg)

    hydra.utils.log.info("Instantiating <WandbLogger>")
    if cfg.logging.wandb_key_api:
        os.environ["WANDB_API_KEY"] = cfg.wandb_key_api
    os.environ["WANDB_DIR"] = str(get_project_root())
    wandb_config = cfg.logging.wandb
    wandb_logger = WandbLogger(
        **wandb_config,
        name=os.getcwd().split("/")[-1],
        save_dir=hydra_dir,
        tags=cfg.core.tags,
    )
    hydra.utils.log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
    wandb_logger.watch(
        model,
        log=cfg.logging.wandb_watch.log,
        log_freq=cfg.logging.wandb_watch.log_freq,
    )

    # pl trainer

    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        val_check_interval=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        **cfg.train.pl_trainer,
    )
    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=data_module)

    if use_val_test:
        # perform eval and test on best model
        hydra.utils.log.info("Starting testing!")
        trainer.validate(datamodule=data_module)
        trainer.test(datamodule=data_module)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

    print("done")
    # for epoch in range(config.epochs):


@hydra.main(
    config_path=str(os.path.join(get_project_root(), "config")),
    config_name="default.yaml",
)
def main(cfg: omegaconf.DictConfig):
    hydra_dir = os.getcwd()
    train(cfg, hydra_dir)


if __name__ == "__main__":
    main()
