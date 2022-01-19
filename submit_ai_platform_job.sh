export PROJECT_ID=sensi-anal
export IMAGE_REPO_NAME=bert_training
export IMAGE_TAG=bert_training_cpu
export IMAGE_URI=gcr.io/"$PROJECT_ID"/"$IMAGE_REPO_NAME":"$IMAGE_TAG"

export REGION=us-central1
export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)


gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  logging.wandb_key_api=
