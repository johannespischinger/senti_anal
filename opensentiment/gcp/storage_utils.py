from datetime import datetime
from google.cloud import storage


def save_to_model_gs(save_dir: str, model_name: str) -> None:
    """Function to upload model to google storage bucket"""
    scheme = "gs://"
    bucket_name = save_dir[len(scheme) :].split("/")[0]
    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = save_dir[len(prefix) :].rstrip("/")

    datetime_ = datetime.now().strftime("model_%Y%m%d_%H%M%S")

    name = model_name.split("/")[-3]
    if bucket_path:
        model_path = "{}/{}/{}".format(bucket_path, datetime_, name)
    else:
        model_path = "{}/{}".format(datetime_, name)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model_name)
