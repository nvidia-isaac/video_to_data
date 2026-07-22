FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG PHANTOM_COMMIT=a8bb81c1bbe6ade129a1f6f0906482f510354a5e
ARG PHANTOM_HAMER_COMMIT=7f4a703e49278f2002f831d93aed867ad6977281
ARG VITPOSE_COMMIT=b65caea48277175584e6fd0f3253ed2e34b82e3e
ARG CHUMPY_COMMIT=580566eafc9ac68b2614b64d6f7aaa84eebb70da
ARG MANOTORCH_COMMIT=a2a70c591f91551078b7bb2af9b5d9f275b626e0
LABEL org.opencontainers.image.title="V2D Phantom hand tracker"
LABEL org.opencontainers.image.source="https://github.com/MarionLepert/phantom"
LABEL org.opencontainers.image.revision="${PHANTOM_COMMIT}"
LABEL io.v2d.phantom-hamer.revision="${PHANTOM_HAMER_COMMIT}"
LABEL io.v2d.vitpose.revision-not-executed="${VITPOSE_COMMIT}"
LABEL io.v2d.chumpy.revision="${CHUMPY_COMMIT}"
LABEL io.v2d.manotorch.revision="${MANOTORCH_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYOPENGL_PLATFORM=egl \
    PIP_NO_CACHE_DIR=1
WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg git libegl1 libgl1 libglib2.0-0 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Official Phantom fork, detached at the parent's exact submodule pin. ViTPose
# is deliberately not initialized: TACO handedness is resolved by the explicit
# bimanual policy in this image, and its 3.3 GB checkpoint is not executed.
RUN git init /opt/phantom-hamer \
    && git -C /opt/phantom-hamer remote add origin https://github.com/MarionLepert/phantom-hamer.git \
    && git -C /opt/phantom-hamer fetch --depth 1 origin ${PHANTOM_HAMER_COMMIT} \
    && git -C /opt/phantom-hamer checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/phantom-hamer rev-parse HEAD)" = "${PHANTOM_HAMER_COMMIT}"

RUN pip install \
        "numpy==1.26.4" \
        "opencv-python-headless==4.11.0.86" \
        "Pillow==11.2.1" \
        "scikit-image==0.25.2" \
        "smplx==0.1.28" \
        "yacs==0.1.8" \
        "pytorch-lightning==2.5.2" \
        "torchmetrics==1.6.1" \
        "timm==1.0.15" \
        "einops==0.8.1" \
        "pyrender==0.1.45" \
        "trimesh==4.6.10" \
        "hydra-core==1.3.2" \
        "omegaconf==2.3.0" \
        "rich==14.0.0" \
        "webdataset==0.2.111" \
        "transformers==4.42.4" \
        "safetensors==0.5.3" \
        "tqdm==4.67.1" \
        "deprecation==2.1.0" \
        "manotorch @ git+https://github.com/lixiny/manotorch.git@${MANOTORCH_COMMIT}" \
        "chumpy @ git+https://github.com/mattloper/chumpy.git@${CHUMPY_COMMIT}" \
    && pip install --no-deps -e /opt/phantom-hamer

RUN mkdir -p /opt/app/inpainting
COPY . /opt/app/inpainting/phantom_tracker
RUN touch /opt/app/inpainting/__init__.py \
    && chmod -R a+rX /opt/app /opt/phantom-hamer \
    && python -c "import chumpy; from hamer.models import HAMER; from manotorch.axislayer import AxisLayerFK; from transformers import AutoModelForZeroShotObjectDetection; print('Phantom imports OK')"

ENV PYTHONPATH=/opt/app:/opt/phantom-hamer \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

CMD ["python", "-m", "inpainting.phantom_tracker.pipeline", "--help"]
