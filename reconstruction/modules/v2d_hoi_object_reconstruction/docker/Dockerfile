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

# Clone a tested, Apache-2.0-licensed 3d-object-reconstruction revision from
# the public GitHub repo (V1). Keep this pinned so release builds do not change
# when upstream main advances.
# To upgrade to V2 (internal dev branch) when released, replace with:
#   ARG GITLAB_TOKEN
#   RUN git clone --branch dev/bundlefusion-replacement-and-texture-refinement --depth 1 \
#       https://oauth2:${GITLAB_TOKEN}@gitlab-master.nvidia.com/metropolis-perf/3d-reconstruction/3d-object-reconstruction-github.git \
#       /workspace/3d-object-reconstruction
ARG OBJECT_RECONSTRUCTION_COMMIT=e7aa98dc2dcc92de9cebb6f2a991fa6d2aa6f284
RUN git init /workspace/3d-object-reconstruction && \
    git -C /workspace/3d-object-reconstruction remote add origin \
        https://github.com/NVIDIA/3DObjectReconstruction.git && \
    git -C /workspace/3d-object-reconstruction fetch --depth 1 origin \
        ${OBJECT_RECONSTRUCTION_COMMIT} && \
    git -C /workspace/3d-object-reconstruction checkout --detach FETCH_HEAD && \
    test "$(git -C /workspace/3d-object-reconstruction rev-parse HEAD)" = \
        "${OBJECT_RECONSTRUCTION_COMMIT}"

# Fix upstream texture-baking clipping in 3d-object-reconstruction.
# The renderer now accepts an optional config-driven zfar. Old configs still get
# the dynamic camera-distance fallback instead of reverting to far*sc_factor.
RUN python3 - <<'EOF'
import pathlib

f = pathlib.Path("/workspace/3d-object-reconstruction/src/nvidia/objectreconstruction/networks/nerf_runner.py")
src = f.read_text()

old = "def build_texture_from_images(mesh, reader, poses, Height, Width, K, sc_factor, far, tex_res, debug_dir=None, config=None,logger: logging.Logger = logger):"
new = "def build_texture_from_images(mesh, reader, poses, Height, Width, K, sc_factor, far, tex_res, debug_dir=None, config=None, texture_zfar=None, texture_zfar_camera_distance_margin=2.0, logger: logging.Logger = logger):"
assert old in src, "pinned build_texture_from_images signature not found in nerf_runner.py"
src = src.replace(old, new, 1)

old = "    renderer = ModelRendererOffscreen([], cam_K=K, H=Height, W=Width, zfar=far*sc_factor)"
new = (
    "    # Use config-driven texture zfar when provided; otherwise compute a safe fallback\n"
    "    # from actual camera distances to avoid clipping the mesh.\n"
    "    cam_dists = np.linalg.norm(poses[:, :3, 3], axis=-1)\n"
    "    if texture_zfar is not None:\n"
    "        computed_zfar = float(texture_zfar)\n"
    "        zfar_source = \"config\"\n"
    "    else:\n"
    "        zfar_margin = float(texture_zfar_camera_distance_margin)\n"
    "        computed_zfar = max(far * sc_factor, float(cam_dists.max()) * zfar_margin)\n"
    "        zfar_source = f\"auto margin={zfar_margin:.3f}\"\n"
    "    logging.info(f\"Texture baking zfar: {computed_zfar:.3f} \"\n"
    "                 f\"({zfar_source}, far={far}, sc_factor={sc_factor:.3f}, \"\n"
    "                 f\"max_cam_dist={cam_dists.max():.3f})\")\n"
    "\n"
    "    renderer = ModelRendererOffscreen([], cam_K=K, H=Height, W=Width, zfar=computed_zfar)"
)
assert old in src, "zfar anchor not found in nerf_runner.py"
src = src.replace(old, new, 1)

f.write_text(src)

f = pathlib.Path("/workspace/3d-object-reconstruction/src/nvidia/objectreconstruction/networks/nvbundlesdf.py")
src = f.read_text()

old = (
    "            config=cfg_texture['color_fusion'] if 'color_fusion' in cfg_texture else None,\n"
    "            logger=self.logger\n"
    "        )"
)
new = (
    "            config=cfg_texture['color_fusion'] if 'color_fusion' in cfg_texture else None,\n"
    "            texture_zfar=self.cfg_nerf.get('texture_zfar'),\n"
    "            texture_zfar_camera_distance_margin=self.cfg_nerf.get('texture_zfar_camera_distance_margin', 2.0),\n"
    "            logger=self.logger\n"
    "        )"
)
assert old in src, "pinned texture zfar call anchor not found in nvbundlesdf.py"
src = src.replace(old, new, 1)

f.write_text(src)
print("3d-object-reconstruction texture zfar patch applied successfully")
EOF

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

# Install BundleSDF CUDA extension. Keep this pinned so release builds do not
# change when upstream nvbundlesdf advances.
ARG BUNDLESDF_COMMIT=56f21198759558f921c5e74e247c7a6dbf611d66
RUN git init /BundleSDF && \
    git -C /BundleSDF remote add origin https://github.com/NVLabs/BundleSDF.git && \
    git -C /BundleSDF fetch --depth 1 origin ${BUNDLESDF_COMMIT} && \
    git -C /BundleSDF checkout --detach FETCH_HEAD && \
    test "$(git -C /BundleSDF rev-parse HEAD)" = "${BUNDLESDF_COMMIT}"
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

# Copy and install shared/runtime packages used by the OSMO HOI workflow.
COPY v2d_common /workspace/v2d_common
COPY v2d_mv /workspace/v2d_mv
COPY v2d_sam2 /workspace/v2d_sam2
COPY v2d_bundlesdf/__init__.py /workspace/v2d_bundlesdf/__init__.py
COPY v2d_bundlesdf/lib /workspace/v2d_bundlesdf/lib
RUN pip install -e "/workspace/v2d_common[io]" \
    -e /workspace/v2d_mv \
    -e /workspace/v2d_sam2/lib \
    -e /workspace/v2d_bundlesdf/lib

# Copy hoi_object_reconstruction library into the image
COPY v2d_hoi_object_reconstruction/__init__.py /workspace/v2d_hoi_object_reconstruction/__init__.py
COPY v2d_hoi_object_reconstruction/lib /workspace/v2d_hoi_object_reconstruction/lib
RUN pip install -e /workspace/v2d_hoi_object_reconstruction/lib
# Fail the image build if a container-only numerical dependency is missing.
# cv2 and ffmpeg are supplied by the image's system/CUDA build rather than pip.
RUN python -c "import importlib; [importlib.import_module(name) for name in (\
'v2d_hoi_object_reconstruction.lib.check_sfm_scan_quality', \
'v2d_hoi_object_reconstruction.lib.convert_poses_to_matrix', \
'v2d_hoi_object_reconstruction.lib.detect_stage1_end', \
'v2d_hoi_object_reconstruction.lib.postprocess_masks', \
'v2d_hoi_object_reconstruction.lib.prepare_FP_folder', \
'v2d_hoi_object_reconstruction.lib.sam3d_srt', \
'v2d_hoi_object_reconstruction.lib.scale_mesh_srt', \
'v2d_hoi_object_reconstruction.lib.select_sam3d_best', \
'v2d_hoi_object_reconstruction.lib.select_sam3d_frames', \
'v2d_hoi_object_reconstruction.lib.center_mesh', \
'v2d_hoi_object_reconstruction.lib.stitch_mp4')]"
ENV PYTHONPATH=/workspace:$PYTHONPATH
