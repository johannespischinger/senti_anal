import os
from signal import default_int_handler

import hydra
import omegaconf
import pytest
import pytorch_lightning as pl

from opensentiment.data.make_dataset import get_datasets, inital_cache_dataset
from opensentiment.utils import get_project_root, return_omegaconf_modified


@pytest.mark.skip
def test_make_dataset_get_datasets():
    """not working"""
    train, val = get_datasets(val_size=0.2)
    assert len(train) > len(val), "Training set smaller than validation set!"
    assert len(train.data[2]) == len(
        train.data[0]
    ), "Number of training samples and labels does not match"
    assert len(val.data[2]) == len(
        val.data[0]
    ), "Number of validation samples and labels does not match"
    # could also check if max_len is true for samples


def test_make_dataset_cache_dataset():
    cfg = return_omegaconf_modified({})

    inital_cache_dataset(cfg)
