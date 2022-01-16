from pathlib import Path
from google.cloud import storage
from datetime import datetime


def get_project_root() -> Path:
    """return Path to the project directory, top folder of opensentiment"""
    return Path(__file__).parent.parent


def save_to_model_gs(save_dir: str, model_name: str) -> None:
    scheme = "gs://"
    bucket_name = save_dir[len(scheme):].split("/")[0]
    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = save_dir[len(prefix):].rstrip("/")

    datetime_ = datetime.now().strftime("model_%Y%m%d_%H%M%S")

    if bucket_path:
        model_path = "{}/{}/{}".format(bucket_path, datetime_, model_name)
    else:
        model_path = "{}/{}".format(datetime_, model_name)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model_name)
