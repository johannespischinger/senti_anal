#! /bash/sh

echo "experiments=$1"
echo "wandb_key_api=$2"
echo "git_tag=$3"

git pull
if [ "$3" != "" ]
then
  echo "checkout {$3}"
  git checkout "$3" data.dvc
else
  echo "checkout v1.0"
  git checkout v1.0 data.dvc
fi

dvc pull
python -u opensentiment/models/train_model.py "experiments=$1" "wandb_key_api=$2"
