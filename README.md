Sentiment Analysis
==============================

Final project for course MLOps course at DTU. 

[![codecov](https://codecov.io/gh/johannespischinger/senti_anal/branch/master/graph/badge.svg?token=CI49NOMH1J)](https://codecov.io/gh/johannespischinger/senti_anal)
![CI pytest](https://github.com/johannespischinger/senti_anal/actions/workflows/python_pip_unittests_lint.yml/badge.svg)
![build-docs](https://github.com/johannespischinger/senti_anal/actions/workflows/build-docs-ghpages.yml/badge.svg)

[Read the docs](https://johannespischinger.github.io/senti_anal/)

Project Description
------------

**_Overall goal of the project:_**

Building and running a sentimental analysis model using a pretrained model "Bert" from the huggingface/transformer 
framework based on the dataset [amazon_polarity](https://huggingface.co/datasets/amazon_polarity).
The dataset contains of about ~35 mio. reviews from Amazon up to March 2013 (in total about 18 years of reviews). 
As a result of the project, the model should analyse new Amazon reviews and classify them either as positive or 
negative rating. 
The overall goal is to learn working with the huggingface/transformer library, applying the various taught 
tools/frameworks from [SkafteNicki/dtu_mlops](https://github.com/SkafteNicki/dtu_mlops) 
to setup a proper ML operations project. 

As already mentioned above, we are using the [Transformer framework](https://github.com/huggingface/transformers) 
to access the pretrained BERT embeddings and to use the preprocessing tools (e.g. tokenizer) for the sentimental analysis. 
The dataset is  directly loaded from the huggingface hub. 
As benchmark model we use the frozen embeddings of BERT and add a final classification layer as proposed in this 
[jupyter notebook](https://github.com/Nitesh0406/-Fine-Tuning-BERT-base-for-Sentiment-Analysis./blob/main/BERT_Sentiment.ipynb).
As a second idea we want to use the already fine-tuned BERT embeddings from another sentiment analysis task from this 
project [fabriceyhc/bert-base-uncased-amazon_polarity](https://huggingface.co/fabriceyhc/bert-base-uncased-amazon_polarity/tree/main).
If training on the normal BERT model takes too long we may consider switching to 
[DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) since it has only 66mio 
parameters compared with 340mio parameters from BERT.

Tools planned (or already implemented) to be used in the project:

| Tools/ Frameworks/<br/>Configurations/ Packages                                                                                     |                      Purpose                      |
|---------------------------------------------------------------------------------------                                              |:-------------------------------------------------:|
| [Conda environement](https://docs.conda.io/en/latest/)                                                                              | Closed environment to facilitate package handling |
| [Wandb](https://wandb.ai/site)                                                                                                      |                Experiment logging                 |
| [Hydra](https://hydra.cc/docs/intro/)                                                                                               |       Managing of config files for training       | 
| [Cookiecutter](https://github.com/cookiecutter/cookiecutter)                                                                        |          Setup the project environement           |
| [black](https://github.com/psf/black/commit/61fe8418cc868723759fb08d76adab1542bb7630) [flake8](https://flake8.pycqa.org/en/latest/) |                   Coding style                    |
| [isort](https://github.com/PyCQA/isort)                                                                                             |                Sorting of imports                 |
|[dvc](https://dvc.org)                                                                                                              |                   Data Versioning                 |






Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## Dev Installation

Default configuration:
```
conda create -y --name py39senti python=3.9 pip
conda activate py39senti
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
pre-commit install
```

CUDA 11.3 configuration
```
# run Default configuration
pip uninstall torch
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## TODOs
### Week 1

- [ ] Create a git repository
- [ ] Make sure that all team members have write access to the github repository
- [ ] Create a dedicated environment for you project to keep track of your packages (using conda)
- [ ] Create the initial file structure using cookiecutter
- [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and 
- [ ] Add a model file and a training script and get that running
- [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [ ] Remember to comply with good coding practices (`pep8`) while doing the project
- [ ] Do a bit of code typing and remember to document essential parts of your code
- [ ] Setup version control for your data or part of your data
- [ ] Construct one or multiple docker files for your code
- [ ] Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [ ] Use wandb to log training progress and other important metrics/artifacts in your code
- [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

- [ ] Write unit tests related to the data part of your code
- [ ] Write unit tests related to model construction
- [ ] Calculate the coverage.
- [ ] Get some continuous integration running on the github repository
- [ ] (optional) Create a new project on `gcp` and invite all group members to it
- [ ] Create a data storage on `gcp` for you data
- [ ] Create a trigger workflow for automatically building your docker images
- [ ] Get your model training on `gcp`
- [ ] Play around with distributed data loading
- [ ] (optional) Play around with distributed model training
- [ ] Play around with quantization and compilation for you trained models

### Week 3

- [ ] Deployed your model locally using TorchServe
- [ ] Checked how robust your model is towards data drifting
- [ ] Deployed your model using `gcp`
- [ ] Monitored the system of your deployed model
- [ ] Monitored the performance of your deployed model

### Additional

- [ ] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Make sure all group members have a understanding about all parts of the project
- [ ] Create a presentation explaining your project
- [ ] Uploaded all your code to github
- [ ] (extra) Implemented pre-commit hooks for your project repository
- [ ] (extra) Used Optuna to run hyperparameter optimization on your model
