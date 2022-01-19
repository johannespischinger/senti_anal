# start this script with sh setup/docker/build_push_docker_gpu.sh 

set -e # exit if sh fails

export PROJECT_ID=sensi-anal
export IMAGE_REPO_NAME=fastapipredict
export IMAGE_TAG=0.0.1
export IMAGE_URI=gcr.io/"$PROJECT_ID"/"$IMAGE_REPO_NAME":"$IMAGE_TAG"

docker build -f setup/docker/fastapipredict.dockerfile -t "$IMAGE_URI" .
# docker push "$IMAGE_URI"