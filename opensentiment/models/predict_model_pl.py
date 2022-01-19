import os
import torch
import pytorch_lightning as pl
from opensentiment.models.bert_model_pl import SentimentClassifierPL
from opensentiment.utils import get_project_root
import hydra
from hydra import compose, initialize, initialize_config_module
import omegaconf
from transformers import AutoModelForSequenceClassification, AutoConfig, BertTokenizer
import pickle


class Prediction:
    def __init__(self, path_to_checkpoint: str = "", path_to_datamodule: str = ""):
        self.model = SentimentClassifierPL.load_from_checkpoint(
            os.path.join(get_project_root(), path_to_checkpoint)
        )

        with open(path_to_datamodule, "rb") as handle:
            self.data_module = pickle.load(handle)

    def predict(self, x: dict = None):
        # x as dict with x['content'] with List of inputs & x['label']=None
        features = self.data_module.convert_to_features(x)
        input_ids, attention_masks, _ = (
            features["input_ids"],
            features["attention_mask"],
            features["labels"],
        )
        dict_class = {0: "Negativ", 1: "Positive"}
        out = self.model.forward(
            torch.LongTensor(input_ids), torch.FloatTensor(attention_masks)
        )[0]
        _, sentiment = torch.max(out, dim=1)

        return [
            (sentence, dict_class[int(x)])
            for sentence, x in zip(x["content"], sentiment)
        ]


if __name__ == "__main__":

    prediction_model = Prediction(
        "models/16-01-06/BERT/14knak32/checkpoints/epoch=1-step=14.ckpt",
        "./models/dataloader.pickle",
    )
    x = {"content": ["This is good", "This is bad"], "label": None}
    prediction_model.predict(x)
