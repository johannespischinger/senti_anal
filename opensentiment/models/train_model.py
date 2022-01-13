import torch
from opensentiment.models.bert_model import SentimentClassifier
import transformers
import numpy as np
from collections import defaultdict
import hydra
from omegaconf import DictConfig
import wandb
import os
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
from typing import Any

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).resolve().parents[2]


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    max_norm: float = 1.0,
) -> [torch.Tensor, np.float]:
    model.train()
    train_loss = []
    correct_pred = 0
    total_pred = 0
    for d in data_loader:
        input_ids = d["input_id"]
        attention_masks = d["attention_mask"]
        targets = d["target"]

        # forward prop
        predictions = model(input_ids, attention_masks)
        loss = criterion(predictions, targets)
        _, pred_classes = torch.max(predictions, dim=1)
        # backprop
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # training loss and number of correct prediction
        train_loss.append(loss.item())
        correct_pred += torch.sum(pred_classes == targets)
        total_pred += targets.shape[0]
    return correct_pred / total_pred, np.mean(train_loss)


def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Any,
) -> [torch.Tensor, float]:
    model.eval()
    eval_loss = []
    correct_pred = 0
    total_pred = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_id"]
            attention_masks = d["attention_mask"]
            targets = d["target"]

            # forward prop
            predictions = model(input_ids, attention_masks)
            loss = criterion(predictions, targets)
            _, pred_classes = torch.max(predictions, dim=1)

            eval_loss.append(loss.item())

            correct_pred += torch.sum(pred_classes == targets)
            total_pred += targets.shape[0]
    return correct_pred / total_pred, np.mean(eval_loss)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(cfg: DictConfig) -> None:
    if cfg.wandb_key_api != "":
        os.environ["WANDB_API_KEY"] = cfg.wandb_key_api
    wandb.init(
        project="BERT",
        entity="senti_anal",
        name=os.getcwd().split("/")[-1],
        job_type="train",
    )
    config = cfg.experiments
    torch.manual_seed(config.seed)

    train_set = torch.load(os.path.join(ROOT_PATH, "data/processed/train_dataset.pt"))
    val_set = torch.load(os.path.join(ROOT_PATH, "data/processed/val_dataset.pt"))

    train_loader = DataLoader(train_set, batch_size=config.batch_size)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    model = SentimentClassifier()
    wandb.watch(model, log_freq=100)

    config.learning_rate = 1e-5
    config.epochs = 1
    total_steps = len(train_loader) * config.epochs

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = transformers.AdamW(
        params=model.parameters(), lr=config.learning_rate, correct_bias=False
    )

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=total_steps,
    )

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(config.epochs):

        # training part
        print(f"epoch : {epoch + 1}/{config.epochs}")
        train_acc, train_loss = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            config.max_norm,
        )
        # validation part
        val_acc, val_loss = eval_model(model, val_loader, criterion)

        # saving training logs
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        logger.info(
            f"train_loss: {train_loss}, train_acc: {train_acc} ,val_loss: {val_loss}, val_acc: {val_acc}"
        )

        # saving model if performance improved
        if val_acc > best_accuracy:
            best_model_name = f"best_model_state_{val_acc:.2}.bin"
            torch.save(model.state_dict(), os.path.join(os.getcwd(), best_model_name))
            best_accuracy = val_acc


if __name__ == "__main__":
    train()
