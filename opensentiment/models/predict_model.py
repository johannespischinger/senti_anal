import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging
import wandb

ROOT_PATH = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def predict(model_name: str, batch_size: int = 64) -> None:
    wandb.init(
        project="BERT",
        entity="senti_anal",
        name=os.getcwd().split("/")[-1],
        job_type="test",
    )

    models_path = os.path.join(ROOT_PATH, "models/runs")
    model = torch.load(os.path.join(models_path, model_name))
    model.eval()
    wandb.watch(model, log_freq=100)

    test_set = torch.load(os.path.join(ROOT_PATH, "data/processed/test_dataset.pt"))
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
    logger.info(f"Final test accuracy: {corr_pred/total_pred}")
