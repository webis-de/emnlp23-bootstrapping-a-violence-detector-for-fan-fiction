FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  wget \
  git \
  curl \
  software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.10 \
  python3.10-distutils

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

ARG PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN pip install --upgrade pip
RUN pip install torch transfomers datasets evaluate click
