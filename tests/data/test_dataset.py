import os
from pathlib import Path

import pytest
import torch
import transformers
from torch.utils.data.dataloader import DataLoader

from opensentiment.data.dataset import AmazonPolarity
from opensentiment.utils import get_project_root


class TestAmazonPolarity:
    def test_dataset(self):
        assert os.path.exists(
            os.path.join(get_project_root(), "tests/dummy_dataset/test_dataset.pt")
        ), "dummy_dataset not existing"

        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
        samples = torch.load(
            os.path.join(get_project_root(), "tests/dummy_dataset/test_dataset.pt")
        )
        dataset = AmazonPolarity(samples.sample[:2], samples.target[:2], tokenizer, 128)
        dataloader_set = next(iter(DataLoader(dataset)))
        assert all(
            i in dataloader_set
            for i in ["input_id", "review", "attention_mask", "target"]
        ), f"missing arributes of dataset {dataloader_set}"
