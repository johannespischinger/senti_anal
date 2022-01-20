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

AVAIL_GPU = 0


@pytest.mark.parametrize(
    "config",
    [
        (return_omegaconf_modified({"model": {"transformer_freeze": False}})),
        (return_omegaconf_modified({"model": {"transformer_freeze": True}})),
        (
            return_omegaconf_modified(
                {
                    "model": {"transformer_freeze": False},
                    "data": {
                        "datamodule": {
                            "model_name_or_path": "distilbert-base-uncased-finetuned-sst-2-english"
                        }
                    },
                }
            )
        ),
    ],
)
def test_model(
    config: omegaconf.OmegaConf,
):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra_dir = os.path.join(get_project_root(), ".cache", "Hydratest")
    os.makedirs(hydra_dir, exist_ok=True)
    model: pl.LightningModule = hydra.utils.instantiate(
        config.model,
        **{
            "model_name_or_path": config.data.datamodule.model_name_or_path,
            "train_batch_size": config.data.datamodule.batch_size.train,
        },
        optim=config.optim,
        data=config.data,
        logging=config.logging,
        _recursive_=False,
    )

    for name, param in model.named_parameters():
        if "classifier" in name or not config.model.transformer_freeze:
            # trainable layers
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    return model
