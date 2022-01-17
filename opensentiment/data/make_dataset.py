"""Code based on https://github.com/Nitesh0406/-Fine-Tuning-BERT-base-for-Sentiment-Analysis./blob/main
/BERT_Sentiment.ipynb """
# -*- coding: utf-8 -*-
import logging
import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import transformers
from datasets import load_dataset

from opensentiment.data.dataset import AmazonPolarity
from opensentiment.utils import get_logger_default, get_project_root


def get_datasets(
    val_size: float = 0.2,
    max_len: int = 128,
    tokenizer_name: str = "bert-base-cased",
) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = get_logger_default(__name__)
    logger.info("Downloading data...")

    training_dataset = load_dataset(
        "amazon_polarity",
        split="train",
        cache_dir=os.path.join(get_project_root(), "data", "raw", "huggingface-cache"),
    )
    train_split = training_dataset.train_test_split(test_size=val_size)
    train_dataset = train_split["train"]
    val_dataset = train_split["test"]

    test_dataset = load_dataset(
        "amazon_polarity", split="test", cache_dir=os.path.join(get_project_root())
    )

    logger.info("Create training dataset...")
    tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_name)
    train_data = AmazonPolarity(
        sample=train_dataset.data[2].to_numpy(),
        target=train_dataset.data[0].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    logger.info("Create validation dataset...")
    val_data = AmazonPolarity(
        sample=val_dataset.data[2].to_numpy(),
        target=val_dataset.data[0].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    logger.info("Create test dataset...")
    test_data = AmazonPolarity(
        sample=test_dataset.data[2].to_numpy(),
        target=test_dataset.data[0].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    torch.save(
        train_data, os.path.join(get_project_root(), "data/processed/train_dataset.pt")
    )
    torch.save(
        val_data, os.path.join(get_project_root(), "data/processed/val_dataset.pt")
    )
    torch.save(
        test_data, os.path.join(get_project_root(), "data/processed/test_dataset.pt")
    )
    logger.info("... datasets successfully created and saved")


@hydra.main(
    config_path=str(os.path.join(get_project_root(), "config")),
    config_name="default.yaml",
)
def inital_cache_dataset(cfg: omegaconf.DictConfig):
    data_module: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    data_module.prepare_data()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    inital_cache_dataset()
    get_datasets()
