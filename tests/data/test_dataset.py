from opensentiment.data.dataset import AmazonPolarity
import transformers
from pathlib import Path
import torch
import os
import pytest
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

        assert (
            hasattr(dataset, "review")
            and hasattr(dataset, "input_id")
            and hasattr(dataset, "attention_mask")
            and hasattr(dataset, "target")
        ), f"missing arributes of dataset {dataset.__dict__}"
        assert dataset.review, "Missing samples in dataset"
        assert dataset.input_id, "Missing input ids in dataset"
        assert (
            dataset.attention_mask
        ), "Missing attention mask for BERT model in dataset"
        assert dataset.target, "Missing labels in dataset"
