# syntax=docker/dockerfile:1.7
#
# Multi-stage image for flash_chord, built from the checked-in uv lock.
#   base  -> system GL/X11 libs + uv + Python 3.11.15 venv
#   deps  -> exact locked runtime + training/test dependencies
#   dev   -> source baked at /workspace, GL via X11
#
# CUDA at runtime comes from the PyPI wheels (warp-lang/newton/mujoco bundle
# CUDA 12) plus the host driver via the NVIDIA Container Toolkit. The CUDA 13
# base only supplies nvcc for source builds. Host driver must be >= 580.
ARG CUDA_TAG=13.0.2-devel-ubuntu24.04
ARG UV_TAG=0.11.21

# Pin the uv binary as its own stage: FROM expands ${UV_TAG} reliably, whereas
# a variable inside `COPY --from=<image:${ARG}>` is not expanded. We copy the
# binary out of this stage in `base` below.
FROM ghcr.io/astral-sh/uv:${UV_TAG} AS uvbin

#####################################################################
# base: system libraries + uv + a Python 3.11.15 virtual environment
#####################################################################
FROM nvidia/cuda:${CUDA_TAG} AS base

ENV DEBIAN_FRONTEND=noninteractive

# OpenGL / EGL / X11 client libs (native + headless GUI), fonts, tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 libopengl0 \
        libglu1-mesa libosmesa6 libglfw3 \
        libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 \
        libxinerama1 libxfixes3 libxxf86vm1 libxkbcommon0 libxcb1 \
        fontconfig libfreetype6 \
        git curl ca-certificates xvfb \
    && rm -rf /var/lib/apt/lists/*

# uv: pinned, copied from the uvbin stage above.
COPY --from=uvbin /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# NVIDIA Container Toolkit: all GPUs + GL/EGL/display capabilities.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Exact CPython 3.11.15 + an empty venv at /opt/venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.11.15 \
    && uv venv --python 3.11.15 "${VIRTUAL_ENV}"

#####################################################################
# deps: install the exact locked dependency set into the venv
#####################################################################
FROM base AS deps

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --active --frozen --extra train --extra dev --no-install-project

#####################################################################
# dev: source baked at /workspace (bind-mount the repo there to override it
#      for live editing); flash_chord imports via PYTHONPATH=/workspace/src
#####################################################################
FROM deps AS dev

WORKDIR /workspace
COPY . /workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --active --frozen --extra train --extra dev \
    && python -c "import hydra, omegaconf; assert hydra.__version__ == '1.3.2'"
ENV PYTHONPATH=/workspace/src
CMD ["bash"]
