import torch
from pathlib import Path
import os
import logging
import transformers
from opensentiment.data.dataset import AmazonPolarity
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).resolve().parents[2]


def truncate_dataset(file, steps: int = 1000) -> Tuple[np.array, np.array]:
    train_set = torch.load(os.path.join(ROOT_PATH, f"data/processed/{file}"))
    return train_set.sample[::steps], train_set.target[::steps]


def create_sample_dataset(
        max_len: int = 128,
        tokenizer_name: str = "bert-base-cased",
) -> None:
    train = truncate_dataset("train_dataset.pt")
    val = truncate_dataset("val_dataset.pt")
    test = truncate_dataset("test_dataset.pt")

    logger.info("Create training dataset...")
    tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_name)
    train_data = AmazonPolarity(
        sample=train[0],
        target=train[1],
        tokenizer=tokenizer,
        max_len=max_len,
    )
    logger.info("Create validation dataset...")
    val_data = AmazonPolarity(
        sample=val[0],
        target=val[1],
        tokenizer=tokenizer,
        max_len=max_len,
    )
    logger.info("Create test dataset...")
    test_data = AmazonPolarity(
        sample=test[0],
        target=test[1],
        tokenizer=tokenizer,
        max_len=max_len,
    )

    save_path = os.path.join(ROOT_PATH, "data/dummy")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving truncated datasets to: {save_path}")
    torch.save(train_data, os.path.join(save_path, "train_dummy.pt"))
    torch.save(val_data, os.path.join(save_path, "val_dummy.pt"))
    torch.save(test_data, os.path.join(save_path, "test_dummy.pt"))


if __name__ == "__main__":
    create_sample_dataset()
