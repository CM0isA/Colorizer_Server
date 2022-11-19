FROM tiangolo/uwsgi-nginx-flask:latest

RUN apk --update add bash nano

RUN apk add --no-cache python3-dev \
    && python3 -m ensurepip \
    && pip3 install --upgrade pip

WORKDIR /app

COPY . /app

ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5555

CMD ["python3", "app.py"]