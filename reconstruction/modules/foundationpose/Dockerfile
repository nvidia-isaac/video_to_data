FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    zip \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y wget bzip2 ca-certificates curl git vim tmux g++ gcc build-essential cmake checkinstall gfortran libjpeg8-dev libtiff5-dev pkg-config yasm libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libv4l-dev libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen libflann-dev libboost-all-dev proj-data libproj-dev libyaml-cpp-dev cmake-curses-gui libzmq3-dev freeglut3-dev


RUN cd / && git clone https://github.com/pybind/pybind11 &&\
    cd pybind11 && git checkout v2.10.0 &&\
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF &&\
    make -j6 && make install


RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&\
    tar xvzf ./eigen-3.4.0.tar.gz &&\
    cd eigen-3.4.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make install

RUN pip install \
    scipy \
    joblib \
    scikit-learn \
    ruamel.yaml \
    trimesh \
    pyyaml \
    opencv-python \
    imageio \
    open3d \
    transformations \
    warp-lang \
    einops \
    kornia \
    pyrender


# Install kaolin
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

# Install Pytorch3D
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"


RUN git clone https://github.com/NVlabs/nvdiffrast /workspace/third_party/nvdiffrast && \
    cd /workspace/third_party/nvdiffrast && \
    pip install setuptools wheel ninja && \ 
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install . --no-build-isolation

ENV OPENCV_IO_ENABLE_OPENEXR=1

RUN pip install \
    scikit-image \
    meshcat \
    webdataset \
    omegaconf \
    pypng \
    roma \
    seaborn \
    opencv-contrib-python \
    openpyxl \
    wandb \
    flask

RUN pip install h5py

# Copy service code
COPY modules/foundationpose/third_party/FoundationPose /workspace/modules/foundationpose/third_party/FoundationPose

RUN cd /workspace/modules/foundationpose/third_party/FoundationPose/mycpp \
    && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11

RUN cd /workspace/modules/foundationpose/third_party/FoundationPose/bundlesdf/mycuda && \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install -e .

RUN pip install celery[redis] redis fast_simplification

# Copy common utilities
COPY modules/common /workspace/modules/common
COPY modules/foundationpose/*.py /workspace/modules/foundationpose
COPY modules/foundationpose/functions /workspace/modules/foundationpose/functions

ENV PYTHONPATH=/workspace