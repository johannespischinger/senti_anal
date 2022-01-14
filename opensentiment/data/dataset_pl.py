import pytorch_lightning as pl
from transformers import BertTokenizerFast
import datasets
from torch.utils.data.dataloader import DataLoader


class AmazonPolarityDataModule(pl.LightningDataModule):
    """
    based on https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html

    Args:
        pl ([type]): [description]

    Returns:
        [type]: [description]
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
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        only_take_every_n_sample: int = 1,
        **kwargs,
    ):
        """[summary]

        Args:
            model_name_or_path (str): [description]
            max_seq_length (int, optional): [description]. Defaults to 128.
            train_batch_size (int, optional): [description]. Defaults to 32.
            eval_batch_size (int, optional): [description]. Defaults to 32.
            only_take_every_n_sample (int, optional): [description]. Defaults to 1.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.only_take_every_n_sample = only_take_every_n_sample

        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name_or_path)

    def setup(self, stage: str):
        batch_size_preprocess = 4096
        self.dataset = datasets.load_dataset("amazon_polarity")

        # TODO: option for small dataset
        # even_dataset = dataset.filter(lambda example, indice: indice % 2 == 0, with_indices=True)
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
        datasets.load_dataset("amazon_polarity")
        BertTokenizerFast.from_pretrained(self.model_name_or_path)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"], batch_size=self.eval_batch_size
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
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
    dm = AmazonPolarityDataModule("bert-base-cased", only_take_every_n_sample=1)
    dm.prepare_data()
    dm.setup("fit")
    print(next(iter(dm.train_dataloader())))
    print(next(iter(dm.train_dataloader())["attention_mask"].shape))
    print(len(iter(dm.train_dataloader())))
