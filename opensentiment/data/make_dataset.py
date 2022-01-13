"""Code based on https://github.com/Nitesh0406/-Fine-Tuning-BERT-base-for-Sentiment-Analysis./blob/main
/BERT_Sentiment.ipynb """
# -*- coding: utf-8 -*-
import logging
from datasets import load_dataset
from opensentiment.data.dataset import AmazonPolarity
import transformers
import os
from pathlib import Path
import torch

ROOT_PATH = Path(__file__).resolve().parents[2]


def get_datasets(
    val_size: float = 0.2,
    max_len: int = 128,
    tokenizer_name: str = "bert-base-cased",
) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading data...")

    training_dataset = load_dataset(
        "amazon_polarity", split="train", cache_dir=os.path.join(ROOT_PATH)
    )
    train_split = training_dataset.train_test_split(test_size=val_size)
    train_dataset = train_split["train"]
    val_dataset = train_split["test"]

    test_dataset = load_dataset(
        "amazon_polarity", split="test", cache_dir=os.path.join(ROOT_PATH)
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

    torch.save(train_data, os.path.join(ROOT_PATH, "data/processed/train_dataset.pt"))
    torch.save(val_data, os.path.join(ROOT_PATH, "data/processed/val_dataset.pt"))
    torch.save(test_data, os.path.join(ROOT_PATH, "data/processed/test_dataset.pt"))
    logger.info("... datasets successfully created and saved")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    get_datasets()
