import os
from opensentiment.utils import get_project_root
import pytorch_lightning as pl
import pytest
import omegaconf
import hydra
from typing import List, Tuple, Union, Dict

from collections import abc


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


def return_omegaconf_modified(modification_data: dict):
    """read the default config of config/data/default.yaml and apply deep modified dict.

    returns:
        omegaconf.Omegaconfig:
    """
    config_data = dict(
        omegaconf.OmegaConf.load(
            os.path.join(get_project_root(), "config", "data", "default.yaml")
        )
    )
    config_data = deep_update(config_data, modification_data)

    config_full = omegaconf.OmegaConf.create({"data": dict(config_data)})
    return config_full


@pytest.mark.parametrize(
    "config,shapes_desired",
    [
        (
            return_omegaconf_modified(
                {
                    "datamodule": {
                        "only_take_every_n_sample": 512,
                        "num_workers": {"train": 0},
                    }
                }
            ),
            [
                ("attention_mask", [32, 128]),
                ("input_ids", [32, 128]),
                ("labels", [32]),
                ("token_type_ids", [32, 128]),
            ],
        ),
        (
            return_omegaconf_modified(
                {
                    "datamodule": {
                        "only_take_every_n_sample": 512,
                        "max_seq_length": 32,
                        "batch_size": {
                            "train": 16,
                            "val": 16,
                            "test": 16,
                        },
                    },
                }
            ),
            [
                ("attention_mask", [16, 32]),
                ("input_ids", [16, 32]),
                ("labels", [16]),
                ("token_type_ids", [16, 32]),
            ],
        ),
    ],
)
def test_dataset(
    config: omegaconf.OmegaConf, shapes_desired: List[Tuple[str, List[int]]]
):
    dm: pl.LightningDataModule = hydra.utils.instantiate(
        config.data.datamodule, _recursive_=False
    )
    dm.prepare_data()
    dm.setup("fit")
    sample = next(iter(dm.train_dataloader()))
    shapes = [(x[0], x[1].shape) for x in sample.items()]

    assert len(shapes) == len(sample)
    for i in shapes_desired:
        assert i[0] in sample, f"{i[0]} not in dataset {sample}"
        assert i[1] == list(
            sample[i[0]].shape
        ), f"shape {i[1]} {i[0]} not in expected but instead {list(sample[i[0]].shape)}"
