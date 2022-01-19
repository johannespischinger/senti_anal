#!/bin/bash

export PROJECT_ID=sensi-anal
export IMAGE_REPO_NAME=bert_training
export IMAGE_TAG=bert_training_cpu
export IMAGE_URI=gcr.io/"$PROJECT_ID"/"$IMAGE_REPO_NAME":"$IMAGE_TAG"

cd ..
docker build --no-cache -f docker/trainer_pl.dockerfile -t "$IMAGE_URI" ./
# docker push "$IMAGE_URI"
