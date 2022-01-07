import torch
from torch import nn
from src.data.make_dataset import get_datasets
from src.models.bert_model import SentimentClassifier
import transformers
import numpy as np
from collections import defaultdict
import hydra
from omegaconf import DictConfig
import wandb
import os
import logging


logger = logging.getLogger(__name__)


def train_model(
    model, data_loader, criterian, optimizer, scheduler, max_norm=1.0
):
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
        loss = criterian(predictions, targets)
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


def eval_model(model, data_loader, criterion):
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

            correct_pred += torch.sum(pred_classes==targets)
            total_pred += targets.shape[0]           
    return correct_pred / total_pred , np.mean(eval_loss)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(cfg: DictConfig) -> None:
    wandb.init(
        project="BERT",
        entity="senti_anal",
        name=os.getcwd().split("/")[-1],
        job_type="train",
    )

    config = cfg.experiments
    torch.manual_seed(config.seed)
    train_set, val_set, test_set = get_datasets()

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    num_classes = 2
    class_names = ["negative", "positive"]
    model = SentimentClassifier()

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
            best_model_name = f"best_model_state_{val_acc}.bin"
            torch.save(model.state_dict(), best_model_name)
            best_accuracy = val_acc


if __name__ == "__main__":
    train()
