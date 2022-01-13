# Base image

FROM python:3.9-slim

# Installs pytorch and torchvision.
RUN apt update && \
  apt install --no-install-recommends -y build-essential gcc && \
  apt clean && \
  apt install -y git && \
  apt-get install -y wget && \
  rm -rf /var/lib.apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -r requirements.txt --no-cache-dir

RUN git clone --branch dev https://github.com/johannespischinger/senti_anal.git

WORKDIR /senti_anal

ENTRYPOINT ["sh", "run_docker.sh"]
# Run dvc

# Sets up the entry point to invoke the training file.



# export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
# export IMAGE_REPO_NAME=mnist_pytorch_custom_container
# export IMAGE_TAG=mnist_pytorch_cpu
# export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# docker build -f Dockerfile -t $IMAGE_URI ./

