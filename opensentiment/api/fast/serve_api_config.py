from opensentiment.utils import get_project_root, paths_to_file_ext
import os


model_path_desired = os.path.join(
    get_project_root(),
    "model_store",
    "distilbert-finetuned-2022-01-20",
    "BERT",
    "m5whu2bt",
    "checkpoints",
    "epoch=4-step=494.ckpt",
)

CONFIG = {
    "MODEL_CPTH_PATH": model_path_desired,
    "FALLBACK_MODEL": ["model_store", "ckpt"],
}
