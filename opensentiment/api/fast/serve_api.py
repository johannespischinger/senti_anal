from typing import Optional

from fastapi import FastAPI

from opensentiment.api.fast.serve_api_helper import score_model

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


@app.get("/")
def read_root():
    return {"Model Serve API": "running"}


@app.get("/serve_single")
def read_item(modelname: str, query_text: str):
    # set up
    args = {}
    return_dict = {
        "modelname": modelname,
        "answer": [],
        "success": False,
        "error": "",
        "query_text": query_text,
    }

    args.update({"query_str": query_text})

    success, answer = score_model(args)
    return_dict.update({"answer": answer, "success": success})
    return return_dict
