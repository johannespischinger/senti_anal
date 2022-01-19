import logging
import os
from typing import Dict, List, Tuple
import subprocess
import pickle
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from opensentiment.utils import get_project_root
from opensentiment.gcp import storage_utils, gcp_settings

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
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run}>."
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
    with open("data_module.pickle", "wb") as handle:
        # save for later usage at prediction
        pickle.dump(data_module, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("data_module.pickle", "rb") as handle:
        # load to ensure object was pickleable
        data_module = pickle.load(handle)
    data_module.prepare_data()
    data_module.setup("fit")

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
        os.environ["WANDB_API_KEY"] = cfg.logging.wandb_key_api

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

    if type(cfg.train.pl_trainer.gpus) == str:
        if "max_available" in cfg.train.pl_trainer.gpus:
            # fix gpu limit
            if torch.cuda.is_available():
                cfg.train.pl_trainer.gpus = -1
            else:
                cfg.train.pl_trainer.gpus = 0
            hydra.utils.log.info(
                f"Configured {cfg.train.pl_trainer.gpus} GPUs from max_available"
            )
        else:
            raise Exception(
                f"Configured {cfg.train.pl_trainer.gpus} needs to be int or str(max_available)"
            )

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

    # upload model to gs
    if gcp_settings.SAVE_TO_GS:
        hydra.utils.log.info(f"Uploading model to gcp bucket: {gcp_settings.GS_BUCKET}")
        subprocess.call(["gsutil", "-m", "cp", "-r", hydra_dir, gcp_settings.GS_BUCKET])

    print("done")


@hydra.main(
    config_path=str(os.path.join(get_project_root(), "config")),
    config_name="default.yaml",
)
def main(cfg: omegaconf.DictConfig):
    hydra_dir = os.getcwd()
    train(cfg, hydra_dir, use_val_test=False)


if __name__ == "__main__":
    main()
