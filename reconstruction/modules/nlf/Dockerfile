FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libegl1-mesa \
    libgles2-mesa \
    libgbm1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Install smplfitter (assuming it's available via pip or needs git install)
RUN pip3 install git+https://github.com/isarandi/smplfitter.git

# Install rendering and detection dependencies
RUN pip3 install pyrender trimesh h5py opencv-python torchvision

# Copy the module and common code
COPY modules/nlf /app/modules/nlf
COPY modules/common /app/modules/common

# Set environment variables
ENV PYTHONPATH=/app
ENV SMPL_MODEL_ROOT=/app/modules/nlf/data/smpl_models
ENV NLF_WEIGHTS_PATH=/app/modules/nlf/data/weights/nlf_l_multi_0.3.2.torchscript

# Command is overridden in docker-compose
CMD ["python3", "modules/nlf/api.py"]

