import logging
from pathlib import Path
from typing import Union
import os
from collections import abc
import omegaconf
from hydra import compose, initialize_config_dir
import hydra


def get_project_root() -> Path:
    """
    return Path to the project directory, top folder of opensentiment
    """
    return Path(__file__).parent.parent.resolve()


def get_logger_default(name: Union[None, str]) -> logging.Logger:
    """
    configure logger


    args:
        name
    returns:
        logger
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)

    # configure logger below
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fmt = logging.Formatter(fmt=log_fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # return modified logger

    return logger


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


def return_omegaconf_modified(
    modification_full: dict = {}, rel_path="config", name_yaml="unittest.yaml"
):
    """read the default config of config/default.yaml and apply deep modified dict.

    returns:
        omegaconf.Omegaconfig:
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(str(os.path.join(get_project_root(), rel_path)))
    config_full = compose(name_yaml)
    config_full = deep_update(dict(config_full), modification_full)

    return omegaconf.OmegaConf.create(config_full)
