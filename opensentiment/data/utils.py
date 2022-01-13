import torch
from pathlib import Path
import os
import logging
import transformers
from opensentiment.data.dataset import AmazonPolarity

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).resolve().parents[2]


def truncate_dataset(path):
    pass


def create_sample_dataset(
        max_len: int = 128,
        tokenizer_name: str = "bert-base-cased",
) -> None:
    train_set = torch.load(os.path.join(ROOT_PATH, "data/processed/train_dataset.pt"))
    train_small_data = train_set.sample[::1000]
    train_small_label = train_set.target[:1000]
    val_set = torch.load(os.path.join(ROOT_PATH, "data/processed/val_dataset.pt"))
    val_small_data = val_set.sample[::1000]
    val_small_label = val_set.target[::1000]
    test_set = torch.load(os.path.join(ROOT_PATH, "data/processed/test_dataset.pt"))
    test_small_data = test_set.sample[::1000]
    test_small_label = test_set.target[::1000]

    logger.info("Create training dataset...")
    tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_name)
    train_data = AmazonPolarity(
        sample=train_small_data,
        target=train_small_label,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    logger.info("Create validation dataset...")
    val_data = AmazonPolarity(
        sample=val_small_data,
        target=val_small_label,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    logger.info("Create test dataset...")
    test_data = AmazonPolarity(
        sample=test_small_data,
        target=test_small_label,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    save_path = os.path.join(ROOT_PATH, "data/dummy")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving dummy datasets to: {save_path}")
    torch.save(train_data, os.path.join(save_path, "train_dummy.pt"))
    torch.save(val_data, os.path.join(save_path, "val_dummy.pt"))
    torch.save(test_data, os.path.join(save_path, "test_dummy.pt"))


if __name__ == "__main__":
    create_sample_dataset()
