import torch
import pytorch_lightning as pl
from opensentiment.models.bert_model_pl import SentimentClassifierPL
from opensentiment.utils import get_project_root
import hydra


class Prediction:
    def __init__(self, path_to_checkpoint: str = ""):
        self.loaded_model = SentimentClassifierPL.load_from_checkpoint(
            path_to_checkpoint
        )
        self.loaded_model.eval()

        self.data_module: pl.LightningDataModule = hydra.utils.instantiate(
            self.loaded_model.data["datamodule"], _recursive_=False
        )

    def predict(self, x: dict):
        """
        x: dict
            x['content']: list of strings
            x["label"]: Any / list of labels, currently not used, dummy required
            x["title"]: list of strings, currently not used
        """
        # x as dict with x['content'] with List of inputs & x['label']=None
        features = self.data_module.convert_to_features(x)
        input_ids, attention_masks = (
            features["input_ids"],
            features["attention_mask"],
        )
        dict_class = {0: "Negativ", 1: "Positive"}
        with torch.no_grad():
            out = self.loaded_model.forward(
                torch.LongTensor(input_ids), torch.FloatTensor(attention_masks)
            )[0]
        out_p = torch.nn.functional.softmax(out, dim=1)
        # out_p = torch.exp(out) # softmax
        _, sentiment = torch.max(out_p, dim=1)

        torch.save_pretrained("./model_store/bert_prediction.pt")
        return [
            (sentence, dict_class[int(x)], out_p[count].tolist())
            for count, (sentence, x) in enumerate(zip(x["content"], sentiment))
        ]


if __name__ == "__main__":

    prediction_model = Prediction(
        str(get_project_root())
        + "/.cache/2022-01-19/17-49-18/BERT/14yg4vjs/checkpoints/epoch=1-step=14.ckpt",
    )
    x = {
        "content": ["This is horrible", "not recommended", "This is perfect bitch", ""],
        "label": None,
    }
    answer = prediction_model.predict(x)
    x = {"content": ["This is very good"], "label": None}
    answer = prediction_model.predict(x)
