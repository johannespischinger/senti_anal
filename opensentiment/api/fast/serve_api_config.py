from opensentiment.utils import get_project_root
import os

CONFIG = {
    "MODEL_CPTH_PATH": os.path.join(
        get_project_root(),
        "model_store",
        "pretrained-distilbert-2022-01-19",
        "BERT",
        "epoch=1-step=14.ckpt",
    )
}
