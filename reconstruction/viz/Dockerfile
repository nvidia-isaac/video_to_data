FROM python:3.11-slim

WORKDIR /app

RUN pip install flask h5py numpy

COPY . /app

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]

