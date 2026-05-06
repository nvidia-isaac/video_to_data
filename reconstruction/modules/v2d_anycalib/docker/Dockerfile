FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_anycalib/lib
