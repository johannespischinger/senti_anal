import torch
import pytorch_lightning as pl
from opensentiment.models.bert_model_pl import SentimentClassifierPL
from opensentiment.utils import get_project_root
import hydra


class Prediction:
    def __init__(self, path_to_checkpoint: str = "", preferred_device: str = "cuda"):
        """
        prediction module

        params
            path_to_checkpoint: os.path to model.ckpt
            preferred_device: "cuda" or "cpu",
                default is "cuda" if torch.cuda.is_available() else "cpu"
                "cpu" uses always cpu
        """
        self.device = torch.device(
            preferred_device if torch.cuda.is_available() else "cpu"
        )
        self.cpu = torch.device("cpu")
        self.loaded_model = SentimentClassifierPL.load_from_checkpoint(
            path_to_checkpoint, map_location=self.device, load_pretrain_weights=False
        )
        self.loaded_model.eval().to(self.device)

        self.data_module: pl.LightningDataModule = hydra.utils.instantiate(
            self.loaded_model.data["datamodule"], _recursive_=False
        )

    def predict(self, x: dict):
        """
        return prediction

        x: dict
            x['content']: list of strings
            x["title"]: list of strings, currently not used
        """
        x.update({"label": None})
        features = self.data_module.convert_to_features(x)
        input_ids, attention_masks = (
            features["input_ids"],
            features["attention_mask"],
        )
        dict_class = {0: "Negativ", 1: "Positive"}
        with torch.no_grad():
            out = self.loaded_model.forward(
                torch.LongTensor(input_ids).to(self.device),
                torch.FloatTensor(attention_masks).to(self.device),
            )[0]
        out = torch.nn.functional.softmax(out, dim=1).to(self.cpu)
        # out_p = torch.exp(out) # softmax
        _, sentiment = torch.max(out, dim=1)

        return [
            (sentence, dict_class[int(x)], out[count].tolist())
            for count, (sentence, x) in enumerate(zip(x["content"], sentiment))
        ]


if __name__ == "__main__":

    prediction_model = Prediction(
        str(get_project_root())
        + "/model_store/distilbert-finetuned-2022-01-20/BERT/m5whu2bt/checkpoints/epoch=4-step=494.ckpt",
    )
    x = {
        "content": ["This is horrible", "not recommended", "This is perfect bitch", ""]
    }
    answer = prediction_model.predict(x)
    x = {"content": ["This is very good"], "label": None}
    answer = prediction_model.predict(x)
    print(answer)
