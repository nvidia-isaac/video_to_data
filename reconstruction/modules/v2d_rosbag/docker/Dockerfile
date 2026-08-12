FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Layer dependency installs by package so edits to unrelated modules do not bust pip cache.
COPY v2d_common /workspace/v2d_common
RUN pip install -e "/workspace/v2d_common[io]"

COPY v2d_mv /workspace/v2d_mv
RUN pip install -e /workspace/v2d_mv

COPY v2d_rosbag/lib /workspace/v2d_rosbag/lib
RUN pip install -e /workspace/v2d_rosbag/lib
