Sentiment Analysis
==============================

Final project for course MLOps course at DTU. 

[![codecov](https://codecov.io/gh/johannespischinger/senti_anal/branch/dev/graph/badge.svg?token=CI49NOMH1J)](https://codecov.io/gh/johannespischinger/senti_anal)![CI pytest](https://github.com/johannespischinger/senti_anal/actions/workflows/python_pip_unittests_lint.yml/badge.svg)
![build-docs](https://github.com/johannespischinger/senti_anal/actions/workflows/build-docs-ghpages.yml/badge.svg)

[Read the docs](https://johannespischinger.github.io/senti_anal/)

Project Description
------------

**_Overall goal of the project:_**

Building and running a sentimental analysis model using a pretrained model "DistilBERT" from the huggingface/transformer 
framework based on the dataset [amazon_polarity](https://huggingface.co/datasets/amazon_polarity).
The dataset contains of about ~35 mio. reviews from Amazon up to March 2013 (in total about 18 years of reviews). 
As a result of the project, the model should analyse new Amazon reviews and classify them either as positive or 
negative rating. 
The overall goal is to learn working with the huggingface/transformer library, applying the various taught 
tools/frameworks from [SkafteNicki/dtu_mlops](https://github.com/SkafteNicki/dtu_mlops) 
to setup a proper ML operations project. 

As already mentioned above, we are using the [Transformer framework](https://github.com/huggingface/transformers) 
to access the pretrained DisttilBERT embeddings and to use the preprocessing tools (e.g. tokenizer) for the sentimental analysis. 
The dataset is  directly loaded from the huggingface hub. 
Initially, we used the frozen embeddings of BERT and add a final classification layer as proposed in this 
[jupyter notebook](https://github.com/Nitesh0406/-Fine-Tuning-BERT-base-for-Sentiment-Analysis./blob/main/BERT_Sentiment.ipynb).
However, as training ended up to be too long with BERT we switched to 
[DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) since it has only 66 mio. 
parameters compared with 340mio parameters from BERT.

Tools planned (or already implemented) to be used in the project:

| Tools/ Frameworks/<br/>Configurations/ Packages                                                                                     |                         Purpose                         |
|-------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------:|
| [Conda environement](https://docs.conda.io/en/latest/)                                                                              |    Closed environment to facilitate package handling    |
| [Wandb](https://wandb.ai/site)                                                                                                      |                   Experiment logging                    |
| [Hydra](https://hydra.cc/docs/intro/)                                                                                               |          Managing of config files for training          | 
| [Cookiecutter](https://github.com/cookiecutter/cookiecutter)                                                                        |             Setup the project environement              |
| [black](https://github.com/psf/black/commit/61fe8418cc868723759fb08d76adab1542bb7630) [flake8](https://flake8.pycqa.org/en/latest/) |                      Coding style                       |
| [isort](https://github.com/PyCQA/isort)                                                                                             |                   Sorting of imports                    |
| [dvc](https://dvc.org)                                                                                                              |                     Data Versioning                     |
| [Google cloud](https://https://cloud.google.com/)                                                                                   |           File storage, training, deployment            |
| [docker](https://docker.com)                                                                                                        | Building train and prediction containers for deployment |
| [fastAPI](https://fastapi.tiangolo.com/)                                                                                            |          Project API for prediction interface           |
| [Huggingface](https://huggingface.co/docs/transformers/index)                                                                                        |              Pretrained model, datamodule               |






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
