#! /bash/sh

echo "experiments=$1"
echo "wandb_key_api=$2"
echo "git_tag=$3"

git pull
if [ "$3" -ne "" ]
then
  echo "checkout {$3}"
  git checkout "$3"
else
  echo "checkout v1.0"
  git checkout v1.0
fi

dvc pull
python -u opensentiment/models/train_model.py "experiments=$1" "wandb_key_api=$2"
