# start this script with sh setup/docker/build_push_docker_gpu.sh 

set -e # exit if sh fails

export PROJECT_ID=sensi-anal
export IMAGE_REPO_NAME=fastapipredict_gpu
export IMAGE_TAG=dk.0.1
export IMAGE_URI=gcr.io/"$PROJECT_ID"/"$IMAGE_REPO_NAME":"$IMAGE_TAG"

docker build -f setup/docker/fastapipredict_gpu.dockerfile -t "$IMAGE_URI" .
docker push "$IMAGE_URI"