FROM python:3.10.4-slim-buster

ENV PYTHONUNBUFFERED 1

RUN apt-get update

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt