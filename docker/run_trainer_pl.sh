#! /bash/sh

echo "experiments=$a"
echo "wandb_key_api=$2"
echo "git_tag=$3"
echo "job_dir_gs=$4"

#git pull
if [ "$3" != "" ]
then
  echo "git checkout {$3}"
  git checkout "$3" data.dvc
else
  echo "git checkout v1.0"
  git checkout v1.0 data.dvc
fi
dvc pull

if [ "$a" != "" ]
then
  export a=exp0
fi
echo "Using $a experiment setting!"
python -u opensentiment/models/train_model_pl.py "experiments=$a" "wandb_key_api=$2" "job_dir_gs=$4"
