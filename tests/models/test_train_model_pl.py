import os
from collections import abc
from typing import List, Tuple

import omegaconf
import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
import hydra
from opensentiment.models import train_model_pl
from opensentiment.utils import get_project_root, return_omegaconf_modified


@pytest.mark.parametrize(
    "config",
    [
        (
            return_omegaconf_modified(
                {
                    "train": {"pl_trainer": {"max_steps": 800, "fast_dev_run": False}},
                    "logging": {"wandb": {"mode": "offline"}},
                }
            )
        )
    ],
)
@pytest.mark.long  # 8 min on gpu, 40min
def test_train(
    config: omegaconf.OmegaConf,
):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra_dir = os.path.join(get_project_root(), ".cache", "Hydratest")
    os.makedirs(hydra_dir, exist_ok=True)
    train_model_pl.train(config, hydra_dir, use_val_test=True)


# @pytest.mark.long  # 4 min on gpu, 40min
@pytest.mark.parametrize(
    "config",
    [
        (
            return_omegaconf_modified(
                {
                    "train": {"pl_trainer": {"fast_dev_run": True, "gpus": 0}},
                    "logging": {"wandb": {"mode": "offline"}},
                }
            )
        )
    ],
)
def test_train2(
    config: omegaconf.OmegaConf,
):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra_dir = os.path.join(get_project_root(), ".cache", "Hydratest2")
    os.makedirs(hydra_dir, exist_ok=True)
    train_model_pl.train(config, hydra_dir, use_val_test=False)
