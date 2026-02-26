# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

##############################################################
# Base Stage
##############################################################

FROM nvcr.io/nvidia/isaac-lab:2.3.1 AS base

# Set working directory
WORKDIR /workspace/robotic_grounding

RUN apt-get update && apt-get install -y \
    curl \
    tmux \
    htop \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

##############################################################
# Copy source code and scripts
##############################################################

COPY pyproject.toml ./
COPY README.md ./

COPY source/ ./source/
COPY scripts/ ./scripts/

##############################################################
# Install Python Dependencies
##############################################################

# Install general dependencies
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    tensordict==0.8.3 \
    datasets==2.19.0 \
    wandb==0.22.2 \
    importlib_metadata \
    plotly \
    viser==1.0.15 \
    dearpygui

# Install chumpy and manotorch (using --no-build-isolation to avoid pip module issues in old setup.py)
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/mattloper/chumpy.git \
    git+https://github.com/lixiny/manotorch.git

# Install Retargeting dependencies
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    mujoco \
    judo-rai>=0.0.5 \
    "pin-pink>=3.5.0,<4" \
    pin==3.7.0 \
    deprecation \
    dearpygui \
    qpsolvers[clarabel,daqp,proxqp,scs]

# FIXME: pytorch3d is not installed with cuda, making the retarget distance_utils module not importable
# # Install pytorch3d (required for retarget distance_utils; --no-build-isolation so build sees Isaac Lab's torch)
# RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir --no-build-isolation \
#     "git+https://github.com/facebookresearch/pytorch3d.git"

# Install Robotic Grounding dependencies
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    onnxruntime-gpu \
    imageio \
    imageio-ffmpeg \
    pytorch-ignite \
    pytest

# Install the project in editable mode
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    -e ./source/robotic_grounding

# Install numpy < 2.0.0 to avoid compatibility issues in Isaac Lab
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    "numpy<2.0.0"

# Create wrapper script for python (for convenience in container shell)
RUN echo '#!/bin/bash' > /usr/local/bin/python && \
    echo "exec ${ISAACLAB_PATH}/isaaclab.sh -p \"\$@\"" >> /usr/local/bin/python && \
    chmod +x /usr/local/bin/python && \
    echo "alias python='${ISAACLAB_PATH}/isaaclab.sh -p'" >> /etc/skel/.bashrc && \
    echo "alias python='${ISAACLAB_PATH}/isaaclab.sh -p'" >> /root/.bashrc

# # Verify pytorch3d is installed (build fails if not importable)
# RUN ${ISAACLAB_PATH}/isaaclab.sh -p -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)"

# Set the default command
CMD ["/bin/bash"]
