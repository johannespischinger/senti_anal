from train_model import eval_model
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT_PATH = Path(__file__).resolve().parents[2]


def predict(batch_size: int = 64):
    # load model

    test_set = torch.load(os.path.join(ROOT_PATH, "data/processed/test_dataset.pt"))
    test_loader = DataLoader(test_set, batch_size=batch_size)
