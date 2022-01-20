#! /bash/sh

dvc pull
python -u opensentiment/models/train_model_pl.py "$@"

