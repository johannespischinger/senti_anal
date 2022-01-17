import os
from collections import abc
from typing import Dict, List, Tuple, Union

import hydra
import omegaconf
import pytest
import pytorch_lightning as pl
from ... import helpers

from opensentiment.utils import get_project_root, return_omegaconf_modified


@pytest.mark.parametrize(
    "config,shapes_desired",
    [
        (
            return_omegaconf_modified(
                {
                    "data": {
                        "datamodule": {
                            "only_take_every_n_sample": 512,
                            "num_workers": {"train": 0},
                        }
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
                    "data": {
                        "datamodule": {
                            "only_take_every_n_sample": 512,
                            "max_seq_length": 32,
                            "batch_size": {
                                "train": 16,
                                "val": 16,
                                "test": 16,
                            },
                        }
                    }
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
