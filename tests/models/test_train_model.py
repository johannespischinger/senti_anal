from opensentiment.models.train_model import train
from opensentiment.models.predict_model import predict
from omegaconf import OmegaConf
import pytest
import os
from pathlib import Path

CONFIG = OmegaConf.create(
    {
        "experiments": {
            "data_path": "tests/dummy_dataset",
            "learning_rate": 1e-05,
            "epochs": 0,
            "batch_size": 64,
            "num_warmup_steps": 0,
            "max_norm": 1.0,
            "seed": 42,
        },
        "wandb_key_api": "",
    }
)
ROOT_PATH = Path(__name__).resolve().parents[2]
DATA_PATH = os.path.join(ROOT_PATH, CONFIG.experiments.data_path)


class TestTraining:
    @pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Data files not found")
    @pytest.mark.order1
    def test_train_model(self):
        history, self.model_name = train(CONFIG)
        assert len(history) == CONFIG.experiments.epochs, "Training not successful!"
        assert (
            history["train_loss"][0] > history["train_loss"][1]
        ), "Training is not decreasing!"

    @pytest.mark.order2
    def test_predict(self):
        acc = predict(self.model_name)
        assert acc <= 1.0, "Accuracy cannot be higher than 100%"
