import os
from datetime import datetime

import pytest

from omegaconf import OmegaConf

from opensentiment.models.predict_model import predict
from opensentiment.models.train_model import train
from opensentiment.utils import get_project_root, return_omegaconf_modified


def test_train_model():
    config = return_omegaconf_modified(
        {}, rel_path="opensentiment/models/config", name_yaml="default_test_config"
    )
    assert os.path.exists(
        os.path.join(get_project_root(), config.experiments.data_path)
    ), f"{os.path.join(get_project_root(), config.experiments.data_path)}"

    datetime_ = datetime.now().strftime("model_%Y%m%d_%H%M%S")
    wk_dir = os.path.join(get_project_root(), ".cache", "runs_test", datetime_)
    os.makedirs(wk_dir, exist_ok=True)
    os.chdir(wk_dir)
    history, model_name = train(config)
    assert len(history) == config.experiments.epochs, "Training not successful!"
    if config.experiments.epochs >= 1:
        assert (
            history["train_loss"][0] >= history["train_loss"][-1]
        ), "Training loss is not decreasing!"

    # test predict
    acc = predict(model_name, wk_dir)
    assert 1.0 >= acc >= 0.0, "Accuracy cannot be higher than 100%"
