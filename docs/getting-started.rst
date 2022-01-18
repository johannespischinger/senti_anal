Getting started
===============

## Dev Installation

Default configuration:
```
conda create -y --name py39senti python=3.9 pip
conda activate py39senti
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
pre-commit install
# verify everything is working
coverage run -m --source=./opensentiment pytest tests -m "not (download or long)"
```

CUDA 11.3 configuration
```
# run Default configuration
pip uninstall torch
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```