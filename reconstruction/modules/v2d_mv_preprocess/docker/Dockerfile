FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY v2d_common /workspace/v2d_common
RUN pip install --no-cache-dir -e /workspace/v2d_common

COPY v2d_mv /workspace/v2d_mv
RUN pip install --no-cache-dir -e "/workspace/v2d_mv[io]"

COPY v2d_mv_preprocess/lib /workspace/v2d_mv_preprocess/lib
RUN pip install --no-cache-dir -e /workspace/v2d_mv_preprocess/lib
