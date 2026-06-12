FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libegl1-mesa \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

# MANO model layers (GPL-3.0) — contained in this image only; never installed
# into the robotic_grounding training image. The MANO .pkl models themselves are
# NOT vendored; they are bind-mounted at runtime via --mano_model_dir.
RUN pip install opencv-python-headless \
                deprecation \
                git+https://github.com/mattloper/chumpy \
                git+https://github.com/lixiny/manotorch

# pytorch3d (used by distance_utils for faster mesh surface sampling) — built
# against the environment torch, so --no-build-isolation. Best-effort: distance_utils
# falls back to trimesh sampling if it's unavailable, so a build-time failure here
# must not fail the image.
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" \
    || echo "WARNING: pytorch3d install failed; distance_utils will use the trimesh fallback"

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_task_library_loader/lib

# robotic_grounding python package (assets excluded) — staged into this module's
# build context by build.py (monorepo: RG lives outside modules/). Provides the
# GPL-clean ManoSharpaData/ManoDex3Data schema + params + quat_from_matrix + naming.
# PYTHONPATH (not pip install) avoids bundling RG's large package_data (assets) into a wheel.
ENV PYTHONPATH=/workspace/v2d_task_library_loader/_rg_pkg
