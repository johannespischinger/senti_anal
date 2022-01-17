import os
from collections import abc
from typing import List, Tuple

import omegaconf
import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir

from opensentiment.models import train_model_pl
from opensentiment.utils import get_project_root


def deep_update(src: dict, overrides: dict):
    """
    Update a nested dictionary or similar mapping.
    Modify ``src`` in place.
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, value in overrides.items():
        if isinstance(value, abc.Mapping) and value:
            returned = deep_update(src.get(key, {}), value)
            src[key] = returned
        else:
            src[key] = overrides[key]
    return src


def return_omegaconf_modified(modification_full: dict = {}):
    """read the default config of config/train/default.yaml + config/data/default.yaml and apply deep modified dict.

    returns:
        omegaconf.Omegaconfig:
    """
    initialize_config_dir(str(os.path.join(get_project_root(), "config")))
    config_full = compose("default.yaml")
    config_full = deep_update(dict(config_full), modification_full)

    return omegaconf.OmegaConf.create(config_full)


@pytest.mark.long
@pytest.mark.parametrize(
    "config",
    [(return_omegaconf_modified({"train": {"pl_trainer": {"max_steps": 800}}}))],
)
def test_dataset(
    config: omegaconf.OmegaConf,
):
    hydra_dir = os.path.join(get_project_root(), ".cache", "Hydratest")
    os.makedirs(hydra_dir, exist_ok=True)
    train_model_pl.train(config, hydra_dir)
