# Build with:
#   docker build -t trainer_pl_gpu:0.0.1 -f setup/docker/trainer_pl_gpu.dockerfile .
# RUN ENTRYPOINT with
#   docker run --entrypoint /bin/python trainer_pl_gpu:0.0.1 -u opensentiment/models/train_model_pl.py $my-hydra-commands
#   e.g. docker run --entrypoint python trainer_pl_gpu:0.0.1 -u opensentiment/models/train_model_pl.py model=trainable logging.wandb_key_api=abcd
#   
FROM python:3.9-slim

# only for debugging purposes!
# RUN git clone --branch dev https://github.com/johannespischinger/senti_anal.git

WORKDIR senti_anal
COPY . .

RUN pip install -r requirements_gpu.txt --no-cache-dir

ENTRYPOINT ["echo", "pass entrypoint like: docker run --entrypoint python trainer_pl_gpu:0.0.1 -u opensentiment/models/train_model_pl.py model=trainable"]