Getting started
===============


## Minimal Installation

Default configuration (Conda 5.10 / Ubuntu 20.04):
```
conda create -y --name py39senti python=3.9 pip
conda activate py39senti

# GPU below
pip install -r requirements.txt
# CUDA 11.3 configuration
# pip install -r requirements_gpu.txt

# git hooks
pre-commit install
# get data
dvc pull
# verify everything is working
coverage run -m --source=./opensentiment pytest tests
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external            <- Data from third party sources.
    │   ├── interim             <- Intermediate data that has been transformed.
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   └── raw                 <- The original, immutable data dump.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── config             <- Source code for use in this project.
    │   ├── data                <- Config files defining the datamodule 
    │   ├── hydra               <- Config files defining hydra setup
    │   ├── logging             <- Config files defining logging in gcp, wandb
    │   ├── model               <- Config files defining used model
    │   ├── optim               <- Config files defining model optimizer
    │   └── train               <- Config files defining train setup (pl.Trainer, metric, early stopping)
    ├── models              <- Folder to store pretrained models locally
    ├── opensentiment      <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   ├── data                <- Script to download or generate data
    │   │   └── make_dataset_pl.py
    │   ├── gcp                 <- Scripts to define settings for google cloud handle
    │   │   └── build_features.py
    │   ├── models              <- Scripts to define and train  model and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model_pl.py
    │   │   └── train_model_pl.py
    │   │   └── bert_model_pl.py
    │   └── api                 <- Scripts to create fastAPI
    ├── setup               <- Files to setup docker (.sh, .yaml, .dockerfile) and pip requirements for cpu and gpu use
    │   ├── docker              <- Folder containing all files to build docker images
    │   ├── pip                 <- Folder containing all files for correct pip setup depening on cpu or gpu
    ├── requirements.txt               <- General requirements file for project
    ├── requirements_gpu.txt               <- Additional requirements file for gpu handling

    └── tox.ini             <- tox file with settings for running tox; see tox.readthedocs.io


--------

