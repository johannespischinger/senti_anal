#!/bin/bash

export PROJECT_ID=sensi-anal
export IMAGE_REPO_NAME=bert_training_2
export IMAGE_TAG=bert_training_cpu
export IMAGE_URI=gcr.io/"$PROJECT_ID"/"$IMAGE_REPO_NAME":"$IMAGE_TAG"

docker build -f trainer.dockerfile -t "$IMAGE_URI" ./
docker push "$IMAGE_URI"
