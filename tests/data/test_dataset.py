from opensentiment.data.dataset import AmazonPolarity
import transformers
from pathlib import Path
import torch
import os

ROOT_PATH = Path(__name__).resolve().parents[2]


class TestAmazonPolarity:
    def test_dataset(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(
            tokenizer_name="bert-base-cased"
        )
        samples = torch.load(
            os.path.join(ROOT_PATH, "tests/dummy_dataset/test_dataset.pt")
        )
        dataset = AmazonPolarity(samples.sample[:2], samples.target[:2], tokenizer, 128)

        assert dataset.review, "Missing samples in dataset"
        assert dataset.input_id, "Missing input ids in dataset"
        assert (
            dataset.attention_mask
        ), "Missing attention mask for BERT model in dataset"
        assert dataset.target, "Missing labels in dataset"
