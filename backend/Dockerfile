FROM python:3.8.7-slim-buster


RUN mkdir /backend
COPY . /backend
WORKDIR /backend


RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

EXPOSE 8000
    
RUN ["uvicorn", "server:app"]