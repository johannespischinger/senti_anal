# Build with:
#   docker build -t trainer_pl_gpu:0.0.1 -f setup/docker/trainer_pl_gpu.dockerfile .
# RUN ENTRYPOINT with
#   docker run --entrypoint /bin/python trainer_pl_gpu:0.0.1 -u opensentiment/models/train_model_pl.py $my-hydra-commands
#   e.g. docker run --gpus all  --entrypoint python trainer_pl_gpu:0.0.1 -u opensentiment/models/train_model_pl.py model=trainable logging.wandb_key_api=abcd
#   
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# get docker
RUN apt-get update && \
    apt-get install -y git
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

ENTRYPOINT ["python", "-u"," opensentiment/models/train_model_pl.py"]