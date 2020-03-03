FROM eu.gcr.io/prowlerio-docker/ubuntu-18.04/python-3.7:latest

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ffmpeg \
       libcairo2-dev \
       git \
       tk-dev

RUN python -m pip install tox

ADD . /src

WORKDIR /src

USER ubuntu
