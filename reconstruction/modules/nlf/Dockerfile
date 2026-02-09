FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install smplfitter (assuming it's available via pip or needs git install)
RUN pip3 install git+https://github.com/isarandi/smplfitter.git

# Copy the module and common code
COPY modules/nlf /app/modules/nlf
COPY modules/common /app/modules/common

# Set environment variables
ENV PYTHONPATH=/app
ENV SMPL_MODEL_ROOT=/app/modules/nlf/data/smpl_models
ENV NLF_WEIGHTS_PATH=/app/modules/nlf/data/weights/nlf_l_multi_0.3.2.torchscript

# Command is overridden in docker-compose
CMD ["python3", "modules/nlf/api.py"]

