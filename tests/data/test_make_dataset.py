from opensentiment.data.make_dataset import get_datasets
import pytest


@pytest.mark.download
class TestMakeDataset:
    def test_make_dataset(self):
        train, val = get_datasets(val_size=0.2)
        assert len(train) > len(val), "Training set smaller than validation set!"
        assert len(train.data[2]) == len(
            train.data[0]
        ), "Number of training samples and labels does not match"
        assert len(val.data[2]) == len(
            val.data[0]
        ), "Number of validation samples and labels does not match"
        # could also check if max_len is true for samples
