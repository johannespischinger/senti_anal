#! /bash/sh

cd ..
mkdir "data"
cd data
gsutil cp -r gs://amazon_polarity data
cd ..
python -u opensentiment/models/train_model_pl.py "$@"

