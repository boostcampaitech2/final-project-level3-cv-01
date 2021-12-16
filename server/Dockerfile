FROM python:3.8.7-slim-buster


RUN mkdir /server
COPY . /server
WORKDIR /server


RUN apt-get -y update
RUN apt-get -y upgrade

EXPOSE 6006
    
CMD ["python", "server.py"]