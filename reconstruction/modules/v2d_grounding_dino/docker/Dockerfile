FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO && \
    cd /tmp/GroundingDINO && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    pip install --no-build-isolation . && \
    rm -rf /tmp/GroundingDINO

RUN pip install "transformers==4.27.4"
RUN pip install huggingface-hub requests Pillow opencv-python

COPY . /workspace

RUN pip install -e "/workspace/v2d_common[io]" \
    -e /workspace/v2d_mv \
    -e /workspace/v2d_grounding_dino/lib
