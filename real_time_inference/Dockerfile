FROM python:3.8.7-slim-buster


RUN mkdir /real_time_adress
COPY . /real_time_adress
WORKDIR /real_time_adress


RUN apt-get -y update
RUN apt-get -y upgrade

RUN pip install pip==21.2.4 && \
  pip install -r requirements.txt

EXPOSE 8501
    
CMD ["streamlit", "run", "streamlit-app.py"]