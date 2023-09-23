FROM python:3.10.4-slim-buster

ENV PYTHONUNBUFFERED 1

RUN apt-get update

WORKDIR /app
ADD . /app

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -r requirements.txt