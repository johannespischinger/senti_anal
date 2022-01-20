# Build with:
#   docker build -t trainer_pl_gpu:0.0.1 -f setup/docker/trainer_pl_gpu.dockerfile .
# RUN ENTRYPOINT with
#   docker run --entrypoint /bin/python trainer_pl_gpu:0.0.1 -u opensentiment/models/train_model_pl.py $my-hydra-commands
#   e.g. docker run --gpus all  --entrypoint python trainer_pl_gpu:0.0.1 -u opensentiment/models/train_model_pl.py model=trainable logging.wandb_key_api=abcd
#   
# general GPU setup
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR app
# get git
RUN apt update && \
  apt install --no-install-recommends -y build-essential gcc && \
  apt clean && \
  apt install -y git && \
  apt-get install -y wget && \
  rm -rf /var/lib.apt/lists/*

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration for google sdk

ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# manage venv
RUN python3 -m pip install --user virtualenv
RUN python3 -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# get files
ENV BRANCH=dev
ENV USER=johannespischinger
ENV REPO=senti_anal
ADD https://api.github.com/repos/$USER/$REPO/git/refs/heads/$BRANCH version.json
RUN git clone -b $BRANCH https://github.com/$USER/$REPO.git
WORKDIR senti_anal
# install requirements
RUN pip3 install --no-cache-dir -r requirements_gpu.txt

# [specifics below]

ENTRYPOINT ["python", "-u"," opensentiment/models/train_model_pl.py"]