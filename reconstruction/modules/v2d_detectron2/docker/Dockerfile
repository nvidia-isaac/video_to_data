FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install Detectron2 (requires --no-build-isolation due to fvcore/iopath)
RUN pip install 'detectron2[all] @ git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation

COPY . /workspace

RUN pip install -e "/workspace/v2d_common[io]" \
                -e /workspace/v2d_mv \
                -e /workspace/v2d_detectron2/lib

ENV TORCH_HOME=/tmp/torch_cache
