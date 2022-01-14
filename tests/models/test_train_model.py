from opensentiment.models.train_model import train
from opensentiment.models.predict_model import predict
from omegaconf import OmegaConf
import pytest
import os
from pathlib import Path
from opensentiment.utils import get_project_root

CONFIG = OmegaConf.create(
    {
        "experiments": {
            "data_path": "tests/dummy_dataset",
            "learning_rate": 1e-05,
            "epochs": 2,
            "batch_size": 64,
            "num_warmup_steps": 0,
            "max_norm": 1.0,
            "seed": 42,
        },
        "wandb_key_api": "",
    }
)


class TestTraining:
    @pytest.mark.skip(reason="no way of currently testing this")
    @pytest.mark.long
    @pytest.mark.parametrize(
        "config",
        [
            CONFIG,
        ],
    )
    def test_train_model(self, config):
        assert os.path.exists(
            os.path.join(get_project_root(), config.experiments.data_path)
        )

        history, model_name = train(config)
        assert len(history) == config.experiments.epochs, "Training not successful!"
        if config.experiments.epochs >= 1:
            assert (
                history["train_loss"][0] >= history["train_loss"][-1]
            ), "Training loss is not decreasing!"

        # test predict
        acc = predict(model_name)
        assert acc <= 1.0 and 0.0 <= acc, "Accuracy cannot be higher than 100%"
