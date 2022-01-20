import os

import datasets
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from opensentiment.utils import get_project_root


class AmazonPolarityDataModule(pl.LightningDataModule):
    """
    based on https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
    """

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        **kwargs,
    ):
        """[summary]

        Args:
            model_name_or_path (str): [description]
            dataset_path (str, optional): [huggiingface dataset path parameter. name or os.path to dir]. Defaults to amazon_polarity.
            max_seq_length (int, optional): [description]. Defaults to 128.
            only_take_every_n_sample (int, optional): [description]. Defaults to 1.
        """
        super().__init__()
        self.save_hyperparameters()

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )
        self.init_dataset_loading()

    def init_dataset_loading(self):
        self.dataset_args = {}

        if not hasattr(self, "dataset_path"):
            self.dataset_path = "amazon_polarity"
            self.dataset_args.update(
                {"revision": "d30d25d3dad590dffe2d3004b4b301dd562dd4f2"}
            )
        print(f"using dataset path {self.dataset_path}")
        self.dataset_args.update({"path": self.dataset_path})
        if hasattr(self, "cache_dir"):
            cache_dir = os.path.join(
                *[
                    i if not (i == "__projectroot__") else str(get_project_root())
                    for i in self.cache_dir
                ]
            )  # convert list of str with __projectroot__ to abspath

            self.dataset_args.update(
                {"cache_dir": cache_dir}
            )  # add option where to store
            print(f"retrieve from / download if not exists to {cache_dir}")

    def setup(self, stage: str):
        batch_size_preprocess = 4096
        self.dataset = datasets.load_dataset(**self.dataset_args)

        # get a random val split
        train_val = self.dataset["train"].train_test_split(0.1, shuffle=False)
        self.dataset["train"] = train_val["train"]
        self.dataset["validation"] = train_val["test"]
        self.dataset["test"] = self.dataset["test"]

        for split in self.dataset.keys():
            self.dataset[split] = (
                self.dataset[split]
                .filter(
                    lambda example, indice: indice % self.only_take_every_n_sample == 0,
                    with_indices=True,
                )
                .map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=["label"],
                    batch_size=batch_size_preprocess,
                )
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        """initiate first downloads"""
        processed_data_path = os.path.join(get_project_root(), "data", "processed")
        # if os.path.isfile(os.path.join(processed_data_path, "train_dataset_pl.pt")):

        datasets.load_dataset(**self.dataset_args)
        AutoTokenizer.from_pretrained(self.model_name_or_path)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size["train"],
            num_workers=self.num_workers["train"],
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.batch_size["val"],
                num_workers=self.num_workers["val"],
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.batch_size["val"],
                    num_workers=self.num_workers["val"],
                )
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["test"],
                batch_size=self.batch_size["test"],
                num_workers=self.num_workers["test"],
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.batch_size["test"],
                    num_workers=self.num_workers["test"],
                )
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch, indices=None):
        # TODO:
        # add for example_batch["content"] option add example_batch["title"]
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            example_batch["content"],
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


if __name__ == "__main__":
    dm = AmazonPolarityDataModule()
    dm.prepare_data()
    dm.setup("fit")
    sample = next(iter(dm.train_dataloader()))
    print(sample)
    print([(x[0], x[1].shape) for x in sample.items()])
    print(len(iter(dm.train_dataloader())))
