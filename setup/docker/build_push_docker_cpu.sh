#!/bin/bash

set -e # exit if sh fails

export PROJECT_ID=sensi-anal
export IMAGE_REPO_NAME=training_cpu
export IMAGE_TAG=0.0.1
export IMAGE_URI=gcr.io/"$PROJECT_ID"/"$IMAGE_REPO_NAME":"$IMAGE_TAG"

docker build --no-cache -f setup/docker/trainer_pl_cpu.dockerfile -t "$IMAGE_URI" ./
docker push "$IMAGE_URI"
