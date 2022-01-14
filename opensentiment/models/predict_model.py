import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging
import wandb
from opensentiment.models.bert_model import SentimentClassifier

from opensentiment.utils import get_project_root

logger = logging.getLogger(__name__)


def predict(
    model_name: str,
    models_path: str,
    batch_size: int = 64,
    data_path: str = "tests/dummy_dataset",
) -> float:
    wandb.init(
        project="BERT",
        entity="senti_anal",
        name=os.getcwd().split("/")[-1],
        job_type="test",
    )

    model = SentimentClassifier()
    model.load_state_dict(torch.load(os.path.join(models_path, model_name)))
    model.eval()
    wandb.watch(model, log_freq=100)

    test_set = torch.load(
        os.path.join(get_project_root(), f"{data_path}/test_dataset.pt")
    )
    test_loader = DataLoader(test_set, batch_size=batch_size)

    total_pred = 0
    corr_pred = 0
    with torch.no_grad():
        for data in test_loader:
            input_ids = data["input_id"]
            attention_masks = data["attention_mask"]
            targets = data["target"]

            predictions = model(input_ids, attention_masks)
            _, pred_class = torch.max(predictions, dim=1)
            corr_pred += torch.sum(pred_class == targets)
            total_pred += targets.shape[0]

    wandb.log({"test_acc": corr_pred / total_pred})
    logger.info(f"Final test accuracy: {corr_pred/total_pred:.4}")
    return corr_pred / total_pred
