Senti_anal
==============================

Final project for course MLOps course at DTU. (use of pytorch) 

Project Description
------------

**_Overall goal of the project:_**

Building and running a sentimental analysis model using a pretrained model "Bert" from the huggingface/transformer framework and the dataset "amazon_polarity".
The dataset contains of about ~35 mio. reviews Amazon up to March 2013 (in total about 18 years of reviews). 
The overall goal is to learn working with the huggingface/transformer library, applying the various taught tools/frameworks from [SkafteNicki/dtu_mlops](https://github.com/SkafteNicki/dtu_mlops) to setup a proper ML operations project. 
As a result of the project, the model should analyse new Amazon reviews and classify them either as positive or negative rating. 


**_What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics):_**

As already mentioned above, this project is using the [Transformer framework](https://github.com/huggingface/transformers) to access the pretrained NLP model for the sentimental analysis.


**_How to you intend to include the framework into your project:_**

The project intents to directly access a pretrained model from [fabriceyhc/bert-base-uncased-amazon_polarity](https://huggingface.co/fabriceyhc/bert-base-uncased-amazon_polarity/tree/main). This way the usage of a classifier and tokenizer is already done in advance. 
In addition, other tools and frameworks support the setup and the configuration of the project:
    
| Tools/ Frameworks/<br/>Configurations/ Packages                                       |                      Purpose                      |
|---------------------------------------------------------------------------------------|:-------------------------------------------------:|
| [Conda environement](https://docs.conda.io/en/latest/)                                | Closed environment to facilitate package handling |
| [Wandb](https://wandb.ai/site)                                                        |                Experiment logging                 |
| [Hydra](https://hydra.cc/docs/intro/)                                                 |       Managing of config files for training       | 
| [Cookiecutter](https://github.com/cookiecutter/cookiecutter)                          |          Setup the project environement           |
| [black](https://github.com/psf/black/commit/61fe8418cc868723759fb08d76adab1542bb7630) |                   Coding style                    |
| [isort](https://github.com/PyCQA/isort)                                               |                Sorting of imports                 |
|tbd |                                                   |


**_What data are you going to run on (initially, may change):_**

At the moment, the idea is to use the dataset [amazon_polarity](https://huggingface.co/datasets/amazon_polarity) to either classify new ratings as positive or negative.


**_What deep learning models do you expect to use:_**

At the moment, this project is using a pretrained version of the BERT model. 


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
