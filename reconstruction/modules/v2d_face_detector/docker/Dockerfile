FROM python:3.11-slim

WORKDIR /app

COPY v2d_common /workspace/v2d_common
RUN pip install -e "/workspace/v2d_common[io]"

COPY v2d_mv /workspace/v2d_mv
RUN pip install -e /workspace/v2d_mv

COPY v2d_face_detector/lib /workspace/v2d_face_detector/lib
RUN pip install -e /workspace/v2d_face_detector/lib
