FROM python:3.9-slim

COPY requirements.txt requirements.txt
COPY opensentiment opensentiment
COPY setup setup
COPY model_store model_store
COPY setup.py setup.py

RUN pip install --no-cache-dir  -r requirements.txt 

CMD ["uvicorn", "opensentiment.api.fast.serve_api:app", "--host", "0.0.0.0", "--port", "80"]

# docker build -t gcr.io/sensi-anal/fastapipredict:0.0.1 .

# docker run -it -p 80:80 gcr.io/sensi-anal/fastapipredict:0.0.1

# docker push gcr.io/sensi-anal/fastapipredict:0.0.1