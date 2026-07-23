FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9

ARG PYTORCH3D_REF=4daa00b41c52455440b938d1b676e00935b204d7
ARG NVDIFFRAST_REF=253ac4fcea7de5f396371124af597e6cc957bfae

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    CMAKE_CUDA_ARCHITECTURES=120 \
    MAX_JOBS=4 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

WORKDIR /app

# FoundationPose's native code needs only Eigen, two Boost components,
# pybind/CMake, and the runtime GL libraries used by Open3D/pyrender.  Avoid the
# original broad multimedia/development install: OpenCV and image I/O arrive as
# Python wheels below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libboost-program-options-dev \
        libboost-system-dev \
        libegl1-mesa \
        libeigen3-dev \
        libgbm1 \
        libgl1 \
        libgles2-mesa \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
        ninja \
        pybind11==2.12.0 \
        setuptools \
        wheel

# Runtime imports are unusually broad in the vendored FoundationPose Utils.py;
# keep its established dependency set while avoiding duplicate OpenCV wheels.
RUN python -m pip install \
        einops \
        fast_simplification \
        gdown \
        h5py \
        imageio \
        joblib \
        kornia \
        meshcat \
        omegaconf \
        open3d \
        openpyxl \
        opencv-contrib-python \
        pypng \
        pyrender \
        pyyaml \
        roma \
        ruamel.yaml \
        scikit-image \
        scikit-learn \
        scipy \
        seaborn \
        transformations \
        trimesh \
        wandb \
        warp-lang \
        webdataset

# NVIDIA publishes an exact cp311 wheel for this Torch/CUDA pair.  Its native
# extensions include Blackwell when used with the cu128 index.
RUN python -m pip install \
        kaolin==0.18.0 \
        --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html

# Both projects contain native CUDA code.  Immutable refs plus the global
# sm_120/FORCE_CUDA settings make the resulting image reproducible and prevent
# a CPU-only build on the GPU-less Docker builder.
RUN python -m pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

RUN python -m pip install --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_REF}"

# Copy only the four packages required by this service.  In particular, do not
# duplicate the full reconstruction/modules build context into the image.
COPY v2d_common /workspace/v2d_common
COPY v2d_mv /workspace/v2d_mv
COPY v2d_mesh /workspace/v2d_mesh
COPY v2d_foundation_pose/lib /workspace/v2d_foundation_pose/lib

RUN cmake \
        -S /workspace/v2d_foundation_pose/lib/FoundationPose/mycpp \
        -B /workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="$(command -v python)" \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build \
        /workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build \
        --parallel "${MAX_JOBS}"

RUN cd /workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda \
    && python -m pip install --no-build-isolation -e .

RUN python -m pip install \
        -e "/workspace/v2d_common[io]" \
        -e "/workspace/v2d_mv[math]" \
        -e /workspace/v2d_mesh/lib \
        -e /workspace/v2d_foundation_pose/lib \
    && mkdir -p "${TORCH_EXTENSIONS_DIR}" \
    && chmod 1777 "${TORCH_EXTENSIONS_DIR}" \
    && chmod -R a+rX /workspace
