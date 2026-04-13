FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG CMAKE_VERSION_MAJOR=3
ARG CMAKE_VERSION_MINOR=25
ARG CMAKE_VERSION_PATCH=3

ARG EIGEN_VERSION_MAJOR=3
ARG EIGEN_VERSION_MINOR=4
ARG EIGEN_VERSION_PATCH=0

ARG OPENCV_VERSION_MAJOR=4
ARG OPENCV_VERSION_MINOR=11
ARG OPENCV_VERSION_PATCH=0

ARG PCL_VERSION_MAJOR=1
ARG PCL_VERSION_MINOR=10
ARG PCL_VERSION_PATCH=0

ARG PYBIND11_VERSION_MAJOR=2
ARG PYBIND11_VERSION_MINOR=13
ARG PYBIND11_VERSION_PATCH=0

ARG YAML_CPP_VERSION_MAJOR=0
ARG YAML_CPP_VERSION_MINOR=8
ARG YAML_CPP_VERSION_PATCH=0

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libblas-dev \
    libssl-dev \
    liblapack-dev \
    gfortran \
    gnupg \
    software-properties-common \
    libflann-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libzmq3-dev \
    ffmpeg \
    libsuitesparse-dev \
    libvtk9-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN cd / && \
    wget http://www.cmake.org/files/v${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}/cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz && \
    tar xf cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz && \
    cd cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH} && \
    ./configure && make && make install && \
    cd / && rm -rf cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}*

# Install Eigen
RUN cd / && \
    wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}/eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    tar xf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    cd eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH} && \
    mkdir build && cd build && cmake .. && make install && \
    cd / && rm -rf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}*

RUN pip3 install -U torch==2.6.0 torchvision==0.21.0

# Install OpenCV with CUDA
RUN cd / && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv_contrib && \
    mkdir -p /opencv/build && cd /opencv/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CUDA_STUBS=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_MATLAB=OFF \
        -DCUDA_FAST_MATH=ON \
        -DMKL_WITH_OPENMP=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_OPENMP=ON \
        -DWITH_QT=OFF \
        -DWITH_OPENEXR=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_FFMPEG=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DENABLE_CXX11=OFF \
        -DBUILD_opencv_xfeatures2d=OFF \
        -DOPENCV_DNN_OPENCL=OFF \
        -DWITH_CUDA=ON \
        -DWITH_OPENCL=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DOPENCV_CUDA_OPTIONS_opencv_test_cudev=-std=c++17 \
        -DCUDA_ARCH_BIN="7.0 7.5 8.0 8.6 9.0" \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DINSTALL_PKGCONFIG=ON \
        -DOPENCV_GENERATE_PKGCONFIG=ON && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /opencv /opencv_contrib

# Install PCL
RUN cd / && \
    git clone --depth 1 --branch pcl-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}.${PCL_VERSION_PATCH} https://github.com/PointCloudLibrary/pcl && \
    mkdir -p /pcl/build && cd /pcl/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_apps=OFF -DBUILD_GPU=OFF -DBUILD_CUDA=OFF \
        -DBUILD_examples=OFF -DBUILD_global_tests=OFF \
        -DBUILD_simulation=OFF -DCUDA_BUILD_EMULATION=OFF \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DPCL_ENABLE_SSE=ON -DPCL_SHARED_LIBS=ON \
        -DWITH_VTK=OFF -DPCL_ONLY_CORE_POINT_TYPES=ON && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /pcl

# Install Pybind11
RUN cd / && \
    git clone --depth 1 --branch v${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH} https://github.com/pybind/pybind11 && \
    mkdir -p /pybind11/build && cd /pybind11/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /pybind11

# Install YAML-CPP
RUN cd / && \
    git clone --depth 1 --branch ${YAML_CPP_VERSION_MAJOR}.${YAML_CPP_VERSION_MINOR}.${YAML_CPP_VERSION_PATCH} https://github.com/jbeder/yaml-cpp && \
    mkdir -p /yaml-cpp/build && cd /yaml-cpp/build && \
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
        -DYAML_CPP_BUILD_TESTS=OFF -DYAML_BUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /yaml-cpp

# Clone 3d-object-reconstruction from the public GitHub repo (V1).
# To upgrade to V2 (internal dev branch) when released, replace with:
#   ARG GITLAB_TOKEN
#   RUN git clone --branch dev/bundlefusion-replacement-and-texture-refinement --depth 1 \
#       https://oauth2:${GITLAB_TOKEN}@gitlab-master.nvidia.com/metropolis-perf/3d-reconstruction/3d-object-reconstruction-github.git \
#       /workspace/3d-object-reconstruction
RUN git clone \
        --branch main \
        --depth 1 \
        https://github.com/NVIDIA/3DObjectReconstruction.git \
        /workspace/3d-object-reconstruction

# Install Python dependencies from repo requirements
RUN pip3 install --no-cache-dir packaging setuptools wheel
RUN pip3 install --no-cache-dir -r /workspace/3d-object-reconstruction/src/requirements.txt

# Install additional Python packages
RUN pip3 install -U pip && pip3 install --no-cache-dir \
    scikit-learn scikit-image --force-reinstall \
    && pip3 install --no-cache-dir --ignore-installed open3d \
    && pip3 install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html \
    && pip3 install --no-cache-dir --no-index pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html \
    && pip3 install --no-cache-dir numpy==1.26.4 scipy joblib scikit-learn scikit-image --force-reinstall \
    && pip3 install --no-cache-dir pyrender \
    && pip3 install --no-cache-dir scikit-sparse==0.4.12 \
    && pip3 install --no-cache-dir theseus-ai pyyaml

# Setup Python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install BundleSDF CUDA extension
RUN git clone --branch nvbundlesdf --depth 1 https://github.com/NVLabs/BundleSDF.git /BundleSDF
RUN cp -r /BundleSDF/mycuda /customize_cuda && \
    cd /customize_cuda && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 pip install . --no-build-isolation

# Build and install BundleTrack
RUN cp -r /BundleSDF/BundleTrack /tmp/BundleTrack && \
    cd /tmp/BundleTrack && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    cp my_cpp*.so /usr/local/lib/python3.10/dist-packages/ && \
    cp libBundleTrack.so /usr/local/lib/ && \
    cp libMY_CUDA_LIB.so /usr/local/lib/ && \
    ldconfig && \
    cd / && rm -rf /tmp/BundleTrack /BundleSDF

# Install FoundationStereo (used for depth in the pipeline)
RUN git clone https://github.com/NVlabs/FoundationStereo.git /FoundationStereo
ENV PYTHONPATH=/FoundationStereo/core:$PYTHONPATH

# Install SAM2
RUN git clone https://github.com/facebookresearch/sam2.git /sam2 && \
    cd /sam2 && \
    SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]" && \
    python3 setup.py build_ext --inplace

# Install RoMa
RUN git clone https://github.com/Parskatt/RoMa.git /RoMa && \
    cd /RoMa && pip3 install . && \
    cd / && rm -rf /RoMa

# Install the 3d-object-reconstruction package
WORKDIR /workspace/3d-object-reconstruction
RUN pip3 install -e src/

# Final cleanup
RUN ldconfig && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/*

ENV PYTHONPATH=/workspace/3d-object-reconstruction/src:$PYTHONPATH

WORKDIR /workspace

# Copy and install shared packages
COPY v2d_common /workspace/v2d_common
RUN pip install -e /workspace/v2d_common

# Copy and install bundlesdf library
COPY v2d_bundlesdf/__init__.py /workspace/v2d_bundlesdf/__init__.py
COPY v2d_bundlesdf/lib /workspace/v2d_bundlesdf/lib
RUN pip install -e /workspace/v2d_bundlesdf/lib

ENV PYTHONPATH=/workspace:$PYTHONPATH
