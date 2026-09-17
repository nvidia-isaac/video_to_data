FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific
WORKDIR /app

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update --fix-missing && apt-get install -y python3-dev git wget curl zip unzip ffmpeg g++ gcc build-essential cmake gfortran pkg-config yasm libgl1 libglib2.0-0 libgtk2.0-dev libjpeg8-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libv4l-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev libflann-dev libboost-all-dev proj-data libproj-dev libyaml-cpp-dev libzmq3-dev freeglut3-dev libxrender1 libxext6 libsm6 libx11-6 libegl1-mesa libegl1-mesa-dev libgles2-mesa libgbm1 libosmesa6-dev libspatialindex-dev && rm -rf /var/lib/apt/lists/*

RUN cd / && git clone --depth=1 --branch v2.10.0 https://github.com/pybind/pybind11 && cd pybind11 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && make -j6 && make install
RUN cd / && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && tar xzf eigen-3.4.0.tar.gz && cd eigen-3.4.0 && mkdir build && cd build && cmake .. && make install

RUN pip install numpy==1.26.3 scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio imageio-ffmpeg "imageio[pyav]" open3d transformations warp-lang einops kornia pyrender av h5py hdf5plugin videoio==0.3.0 scikit-image meshcat webdataset omegaconf pypng roma seaborn opencv-contrib-python openpyxl wandb matplotlib tqdm pillow accelerate timm yacs smplx tensorboardx pycocotools rtree xatlas
RUN pip install pytorch-lightning hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils networkx==3.2.1 rich dill pandas jsonlines loguru optree fvcore cython py-soma-x==0.2.1
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
RUN git clone https://github.com/NVlabs/nvdiffrast /workspace/third_party/nvdiffrast && cd /workspace/third_party/nvdiffrast && pip install setuptools wheel ninja && TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install . --no-build-isolation
RUN pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" && pip install "git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030"

RUN apt-get update && apt-get install -y libnvinfer-bin=10.7.0.23-1+cuda12.6 libnvinfer10=10.7.0.23-1+cuda12.6 libnvinfer-lean10=10.7.0.23-1+cuda12.6 libnvinfer-plugin10=10.7.0.23-1+cuda12.6 libnvinfer-vc-plugin10=10.7.0.23-1+cuda12.6 libnvinfer-dispatch10=10.7.0.23-1+cuda12.6 libnvonnxparsers10=10.7.0.23-1+cuda12.6 && rm -rf /var/lib/apt/lists/*
RUN pip install fast_simplification gdown "tensorrt-cu12==10.7.0.post1"

COPY v2d_foundation_pose/lib/FoundationPose/mycpp /workspace/v2d_foundation_pose/lib/FoundationPose/mycpp
COPY v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda /workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda
RUN cd /workspace/v2d_foundation_pose/lib/FoundationPose/mycpp && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11
RUN cd /workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda && TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install -e .

COPY v2d_common /workspace/v2d_common
COPY v2d_mv /workspace/v2d_mv
COPY v2d_mesh/lib /workspace/v2d_mesh/lib
COPY v2d_foundation_pose/lib /workspace/v2d_foundation_pose/lib
COPY v2d_moge/lib /workspace/v2d_moge/lib
COPY v2d_sam3d_body /workspace/v2d_sam3d_body
COPY v2d_cari4d/lib /workspace/v2d_cari4d/lib
ENV PIP_CONSTRAINT=/workspace/v2d_cari4d/lib/pip-constraints.txt
RUN pip install -e "/workspace/v2d_common[io]" -e "/workspace/v2d_mv[math,vis]" -e /workspace/v2d_mesh/lib -e /workspace/v2d_foundation_pose/lib -e /workspace/v2d_moge/lib -e /workspace/v2d_sam3d_body/lib -e /workspace/v2d_cari4d/lib
RUN pip install pytest==9.1.1

ENV PYTHONPATH=/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYOPENGL_PLATFORM=egl
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/v2d_cari4d/lib/cari4d
