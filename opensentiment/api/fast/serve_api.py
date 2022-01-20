from fastapi import FastAPI
from fastapi.logger import logger
from opensentiment.models.predict_model_pl import Prediction
from opensentiment.api.fast.serve_api_config import CONFIG
from typing import List

app = FastAPI(
    title="Model Serve API",
    description="Serving models of the senti anal project",
    version="0.0.1",
    terms_of_service="https://github.com/johannespischinger/senti_anal",
    contact={
        "name": "Michael Feil, Johannes Pischinger, Max Frantzen",
        "url": "https://github.com/johannespischinger/senti_anal",
        "email": "",
    },
    license_info={
        "name": "MIT",
        "url": "https://raw.githubusercontent.com/git/git-scm.com/main/MIT-LICENSE.txt",
    },
)


# FastAPI app stuff:
@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and load model
    """

    logger.info("Running envirnoment: {}".format(CONFIG["MODEL_CPTH_PATH"]))

    # Initialize the pytorch model prediction
    model_predict = Prediction(CONFIG["MODEL_CPTH_PATH"])
    # add model
    app.package = {"model_predict": model_predict}


@app.get("/")
def read_root():
    return {"Model Serve API": "running"}


@app.get("/api/v1/serve_single")
def inference_single(query_text: str):
    # set up
    prediction_input = {"content": [query_text], "label": None}
    prediction = app.package["model_predict"].predict(prediction_input)

    return {
        "prediction": prediction,
        "query_text": query_text,
    }


@app.get("/api/v1/serve_batch")
def inference_batch(query_list: List[str]):
    """
    {
        "query_list": [
            "review 1",
            "review 2"
        ]
    }
    """

    # set up
    prediction_input = {"content": query_list, "label": None}
    prediction = app.package["model_predict"].predict(prediction_input)

    return {
        "prediction": prediction,
        "query_list": query_list,
    }
