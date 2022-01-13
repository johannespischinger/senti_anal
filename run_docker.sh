#! /bash/sh

echo "experiments: $1"
echo "wandb_key_api: $2"

dvc pull
python -u opensentiment/models/train_model.py $1 $2