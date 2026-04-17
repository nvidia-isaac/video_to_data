FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install --no-cache-dir \
    -e /workspace/v2d_common \
    -e /workspace/v2d_mv \
    -e /workspace/v2d_rosbag/lib
