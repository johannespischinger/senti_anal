from opensentiment.data.dataset_pl import AmazonPolarityDataModule
import torch
import pytest


@pytest.mark.parametrize(
    "module_inputs,shapes_desired",
    [
        (
            {"only_take_every_n_sample": 512},
            [
                ("attention_mask", [32, 128]),
                ("input_ids", [32, 128]),
                ("labels", [32]),
                ("token_type_ids", [32, 128]),
            ],
        ),
        (
            {
                "only_take_every_n_sample": 512,
                "train_batch_size": 16,
                "max_seq_length": 8,
            },
            [
                ("attention_mask", [16, 8]),
                ("input_ids", [16, 8]),
                ("labels", [16]),
                ("token_type_ids", [16, 8]),
            ],
        ),
    ],
)
def test_dataset(module_inputs, shapes_desired):
    dm = AmazonPolarityDataModule("bert-base-cased", **module_inputs)
    dm.prepare_data()
    dm.setup("fit")
    sample = next(iter(dm.train_dataloader()))
    shapes = [(x[0], x[1].shape) for x in sample.items()]

    assert len(shapes) == len(sample)
    for i in shapes_desired:
        assert i[0] in sample, f"{i[0]} not in dataset {sample}"
        assert i[1] == list(
            sample[i[0]].shape
        ), f"shape {i[1]} {i[0]} not in expected but instead {list(sample[i[0]].shape)}"
