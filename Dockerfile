# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

EXPOSE 5000

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt


CMD ["gunicorn", "--bind", "0.0.0.0:5000", "{subfolder}.{module_file}:app"]