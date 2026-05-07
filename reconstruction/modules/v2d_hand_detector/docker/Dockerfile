FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# MediaPipe's prebuilt wheels need libGL + a few X libs to load the C runtime.
# Without them, `import mediapipe.solutions` silently degrades — the package
# imports but the `solutions` submodule never registers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy<2" \
    Pillow \
    opencv-python-headless \
    "mediapipe==0.10.18"

# Bake in the Tasks-API HandLandmarker model. Newer than the legacy
# `mediapipe.solutions.hands` model and somewhat more robust on close-up /
# non-frontal hands. Apache 2.0.
RUN mkdir -p /opt/mediapipe_tasks && \
    wget -q -O /opt/mediapipe_tasks/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Sanity-check at build time so a broken install fails the build, not the run.
RUN python -c "import mediapipe as mp; from mediapipe.tasks.python import vision; print('mediapipe', mp.__version__, 'tasks-api OK')"

COPY . /workspace

RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_hand_detector/lib
