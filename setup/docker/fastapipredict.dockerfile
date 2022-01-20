# general GPU setup
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR app
# get git
RUN apt-get update && \
    apt-get install -y git
# manage venv
RUN python3 -m pip install --user virtualenv
RUN python3 -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# get files
ENV BRANCH=dev
ENV USER=johannespischinger
ENV REPO=senti_anal
ADD https://api.github.com/repos/$USER/$REPO/git/refs/heads/$BRANCH version.json
RUN git clone -b $BRANCH https://github.com/$USER/$REPO.git
WORKDIR senti_anal
# install requirements
RUN pip3 install --no-cache-dir -r requirements_gpu.txt

# [specifics below]
RUN git fetch --all --tags
# RUN git checkout tags/3.0
RUN dvc pull

ENTRYPOINT ["uvicorn", "opensentiment.api.fast.serve_api:app", "--host", "0.0.0.0", "--port", "80"]

# docker build -t gcr.io/sensi-anal/fastapipredict:0.0.1 .

# docker run -it -p 80:80 gcr.io/sensi-anal/fastapipredict:0.0.1

# docker push gcr.io/sensi-anal/fastapipredict:0.0.1