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
WORKDIR /workspace/video_to_data/robotic_grounding

RUN apt-get update && apt-get install -y \
    curl \
    tmux \
    htop \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

##############################################################
# Install Python Dependencies (before source copy for caching)
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
    dearpygui \
    scipy \
    scikit-learn

# GPL hand-model deps (chumpy + MANO layers) removed — the Stage-1 dataset load
# (raw -> loaded Parquet via hand FK) moved to reconstruction's
# v2d_task_library_loader. robotic_grounding consumes the loaded Parquet only.

# Install py-soma-x for SOMA body model used by `scripts/retarget/soma_to_g1.py`.
# Assets are auto-downloaded from HuggingFace on first SOMALayer use; the
# robotic_grounding code instantiates SOMALayer with
# `data_root=BODY_MODELS_DIR / "soma"` so per-run assets land at a known path
# inside the bind-mounted repo. Pin via PyPI release.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    py-soma-x

# Install Retargeting dependencies.
# Pin the FULL pinocchio native closure to the last known-good versions.
# pinocchio (`pin`) bundles its native libs via cmeel; leaving them unpinned
# lets a rebuild drift cmeel soversions pinocchio wasn't built against, failing
# at runtime with e.g. "ImportError: liburdfdom_sensor.so.4.0 / libtinyxml2.so.10:
# cannot open shared object file". Pin pin's actual closure (pin -> coal, eigenpy,
# cmeel-*; cmeel-urdfdom -> cmeel-console-bridge, cmeel-tinyxml2). NOTE: include
# `coal` (not `hpp-fcl`) — pinocchio 3.7 depends on coal, and adding hpp-fcl 2.4.4
# makes the resolve impossible (it wants cmeel-boost~=1.83 vs eigenpy's ~=1.87).
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    mujoco \
    judo-rai>=0.0.5 \
    pin==3.7.0 \
    pin-pink==4.2.0 \
    coal==3.0.1 \
    eigenpy==3.10.3 \
    cmeel==0.57.3 \
    cmeel-boost==1.87.0.1 \
    cmeel-urdfdom==4.0.1 \
    cmeel-console-bridge==1.0.2.3 \
    cmeel-tinyxml2==10.0.0 \
    cmeel-tinyxml==2.6.2.3 \
    cmeel-assimp==5.4.3.1 \
    cmeel-octomap==1.10.0 \
    cmeel-qhull==8.0.2.1 \
    cmeel-zlib==1.3.1 \
    usd-core \
    deprecation

# Install QPSolvers
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir --upgrade \
    osqp \
    qpsolvers[clarabel,daqp,proxqp,scs,osqp]==4.9.0

# FIXME: pytorch3d is not installed with cuda, making the retarget distance_utils module not importable
# # Install pytorch3d (required for retarget distance_utils; --no-build-isolation so build sees Isaac Lab's torch)
# RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir --no-build-isolation \
#     "git+https://github.com/facebookresearch/pytorch3d.git"

# Install Robotic Grounding dependencies
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    onnxruntime-gpu \
    imageio \
    imageio-ffmpeg \
    pyrender \
    pytorch-ignite \
    pytest \
    boto3

# Install numpy < 2.0.0 to avoid compatibility issues in Isaac Lab
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    "numpy<2.0.0"

##############################################################
# Copy source code and scripts
##############################################################

COPY pyproject.toml ./
COPY README.md ./

COPY source/ ./source/
COPY scripts/ ./scripts/

# Embed the third-party TACO object meshes the E2E tests need at the canonical
# asset path the pipeline reads from. The fixtures and their CC BY 4.0 LICENSE +
# NOTICE live under ci/assets/taco/meshes/ (© Yun Liu et al.); copy the whole dir
# so the license/attribution travels with the meshes in the image.
COPY ci/assets/taco/meshes/ ./source/robotic_grounding/robotic_grounding/assets/meshes/taco/

# Install the project in editable mode
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    -e ./source/robotic_grounding

# Pre-generate the rigid URDFs for the embedded TACO meshes so the E2E tests that
# load processed sequences (test_train_e2e.py) find them without first running the
# URDF-generation stage. Mesh-only discovery (globs meshes/taco/*_cm.obj); no GPU.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p scripts/generate_rigid_urdfs.py --dataset taco

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
