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
    git \
    git-lfs \
    curl \
    tmux \
    htop \
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
    "pre-commit>=4.3.0,<5"

# Force reinstall pyyaml to fix pre-commit yaml import issue (prebundle location not accessible by git hooks)
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --force-reinstall --no-deps pyyaml

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
    qpsolvers[clarabel,daqp,proxqp,scs]

# Install the project in editable mode
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    -e ./source/robotic_grounding

# Install numpy < 2.0.0 to avoid compatibility issues in Isaac Lab
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    "numpy<2.0.0"

# Create wrapper scripts for python and pre-commit (for convenience in container shell)
RUN echo '#!/bin/bash' > /usr/local/bin/python && \
    echo "exec ${ISAACLAB_PATH}/isaaclab.sh -p \"\$@\"" >> /usr/local/bin/python && \
    chmod +x /usr/local/bin/python && \
    echo '#!/bin/bash' > /usr/local/bin/pre-commit && \
    echo "exec ${ISAACLAB_PATH}/isaaclab.sh -p -m pre_commit \"\$@\"" >> /usr/local/bin/pre-commit && \
    chmod +x /usr/local/bin/pre-commit && \
    echo "alias python='${ISAACLAB_PATH}/isaaclab.sh -p'" >> /etc/skel/.bashrc && \
    echo "alias python='${ISAACLAB_PATH}/isaaclab.sh -p'" >> /root/.bashrc && \
    echo "alias pre-commit='${ISAACLAB_PATH}/isaaclab.sh -p -m pre_commit'" >> /etc/skel/.bashrc && \
    echo "alias pre-commit='${ISAACLAB_PATH}/isaaclab.sh -p -m pre_commit'" >> /root/.bashrc

##############################################################
# Git Configuration
##############################################################

# Initialize git LFS
RUN git lfs install

# Add safe directory for mounted workspace
RUN git config --global --add safe.directory /workspace/robotic_grounding

##############################################################
# Entrypoint Setup
##############################################################

# Create entrypoint script to auto-install pre-commit hooks
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Install pre-commit hooks if .git exists and hooks are not installed' >> /entrypoint.sh && \
    echo 'if [ -d "/workspace/robotic_grounding/.git" ]; then' >> /entrypoint.sh && \
    echo '  if [ ! -f "/workspace/robotic_grounding/.git/hooks/pre-commit" ]; then' >> /entrypoint.sh && \
    echo '    echo "Installing pre-commit hooks..."' >> /entrypoint.sh && \
    echo '    cd /workspace/robotic_grounding && ${ISAACLAB_PATH}/isaaclab.sh -p -m pre_commit install 2>/dev/null || true' >> /entrypoint.sh && \
    echo '  fi' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Set the default command
CMD ["/bin/bash"]
