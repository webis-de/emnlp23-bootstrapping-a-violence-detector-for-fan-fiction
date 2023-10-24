FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

COPY requirements.txt /code/

WORKDIR /code
RUN apt update && apt install git gcc g++ -y && \
    pip install -r requirements.txt && \
    pip uninstall -y small-text && \
    pip install small-text[transformers]==1.0.0b4 && \
    apt-get remove -y git gcc g++

#pip install git+https://github.com/webis-de/small-text.git@stopping-criteria#egg=small-text[transformers] #small-text==1.0.0b4
