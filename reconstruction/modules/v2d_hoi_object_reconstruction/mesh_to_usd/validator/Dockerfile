# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG SIMREADY_FOUNDATION_COMMIT=4d9f3bb2897bc16ff99978774138c68f4e30ecbf
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIMREADY_FOUNDATION_COMMIT=${SIMREADY_FOUNDATION_COMMIT} \
    SIMREADY_FOUNDATION_SPEC_ROOT=/usr/local/lib/python3.12/site-packages/simready/foundation/core

COPY v2d_hoi_object_reconstruction/mesh_to_usd/validator/requirements.txt /tmp/requirements.txt
COPY v2d_hoi_object_reconstruction/mesh_to_usd/validator/build_foundation_requirements.py /tmp/build_foundation_requirements.py
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check \
    && rm /tmp/requirements.txt

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates git; \
    git init /tmp/simready-foundation; \
    git -C /tmp/simready-foundation remote add origin https://github.com/NVIDIA/simready-foundation.git; \
    git -C /tmp/simready-foundation fetch --depth 1 origin "${SIMREADY_FOUNDATION_COMMIT}"; \
    git -C /tmp/simready-foundation checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/simready-foundation rev-parse HEAD)" = "${SIMREADY_FOUNDATION_COMMIT}"; \
    mkdir -p /licenses; \
    cp /tmp/simready-foundation/LICENSE.txt /licenses/simready-foundation-LICENSE.txt; \
    python /tmp/build_foundation_requirements.py \
        --capabilities /tmp/simready-foundation/nv_core/sr_specs/docs/capabilities \
        --output /tmp/simready-foundation/nv_core/sr_specs/_build/python; \
    python -m pip install --no-cache-dir --no-deps /tmp/simready-foundation/nv_core/sr_specs; \
    test -f "${SIMREADY_FOUNDATION_SPEC_ROOT}/capabilities/__init__.py"; \
    test -f "${SIMREADY_FOUNDATION_SPEC_ROOT}/requirements/__init__.py"; \
    apt-get purge -y --auto-remove git; \
    rm -rf /tmp/build_foundation_requirements.py /tmp/simready-foundation /var/lib/apt/lists/*

COPY v2d_hoi_object_reconstruction/mesh_to_usd/validator/validate_usd.py /workspace/mesh_to_usd/validator/validate_usd.py
COPY v2d_hoi_object_reconstruction/mesh_to_usd/validator/specs /workspace/mesh_to_usd/validator/specs
WORKDIR /workspace/mesh_to_usd/validator

USER 65532:65532
ENTRYPOINT ["python", "/workspace/mesh_to_usd/validator/validate_usd.py"]
