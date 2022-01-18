#!/bin/bash

export PROJECT_ID=sensi-anal
export IMAGE_REPO_NAME=bert_training_2
export IMAGE_TAG=bert_training_cpu
export IMAGE_URI=gcr.io/"$PROJECT_ID"/"$IMAGE_REPO_NAME":"$IMAGE_TAG"
export REGION=us-central1
export BUCKET_NAME=model_senti_anal
export JOB_NAME=test4

gcloud config set project sensi-anal
# hydra arguments
export JOB_DIR=gs://model_senti_anal
export WANDB_API_KEY=15389710194bb8f1832918d0696ca145fff8af98
export GIT_TAG=v1.0
export EXPERIMENT=exp0

gcloud ai-platform jobs submit training "$JOB_NAME" \
  --region=$REGION \
  --master-image-uri=$IMAGE_URI \
  -- \
  $EXPERIMENT \
  $WANDB_API_KEY \
  $GIT_TAG \
  $JOB_DIR