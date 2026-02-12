FROM python:3.11-slim

WORKDIR /app

RUN pip install flask h5py numpy opencv-python-headless pillow

COPY modules/common /app/modules/common
COPY viz /app/viz

ENV PYTHONPATH=/app

EXPOSE 8080

ENTRYPOINT ["python", "viz/app.py"]



