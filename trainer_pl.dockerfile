# Base image

FROM python:3.9-slim

# Installs pytorch and torchvision.
RUN apt update && \
  apt install --no-install-recommends -y build-essential gcc && \
  apt clean && \
  apt install -y git && \
  apt-get install -y wget && \
  rm -rf /var/lib.apt/lists/*

# only for debugging purposes!

RUN git clone --branch dev https://github.com/johannespischinger/senti_anal.git
WORKDIR /senti_anal
COPY data data/
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "opensentiment/models/train_model_pl.py"]
