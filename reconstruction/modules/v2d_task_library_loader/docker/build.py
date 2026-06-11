# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import subprocess

IMAGE_NAME = "v2d_task_library_loader"

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir, "..")          # v2d_task_library_loader/
root_dir = os.path.join(module_dir, "..")             # reconstruction/modules/ (build context)
dockerfile_path = os.path.join(current_dir, "Dockerfile")

# robotic_grounding's python package lives outside the modules/ build context
# (monorepo). Stage a copy WITHOUT its ~45G assets dir into the build context so
# the image can import the schema/params/utils; the ~3.5M python package is all
# that's needed. Staged dir is git-ignored; the Dockerfile puts it on PYTHONPATH.
_RG_PKG_SRC = os.path.abspath(
    os.path.join(
        root_dir, "..", "..", "robotic_grounding",
        "source", "robotic_grounding", "robotic_grounding",
    )
)
_RG_STAGE = os.path.join(module_dir, "_rg_pkg", "robotic_grounding")


def stage_robotic_grounding() -> None:
    """Copy RG's python package (NO assets) into the build context.

    Object assets (urdfs/meshes) are NOT baked — the load workflow fetches the
    per-dataset object assets from swift at runtime (kept lean; see the OSMO load
    workflow). human_motion_data (raw) and body_models (MANO, licensed) likewise
    come in at runtime, never in the image.
    """
    stage_root = os.path.join(module_dir, "_rg_pkg")
    if os.path.exists(stage_root):
        shutil.rmtree(stage_root)
    if not os.path.isdir(_RG_PKG_SRC):
        raise FileNotFoundError(
            f"robotic_grounding package not found at {_RG_PKG_SRC}; "
            "expected the monorepo layout video_to_data/{robotic_grounding,reconstruction}."
        )
    shutil.copytree(
        _RG_PKG_SRC,
        _RG_STAGE,
        ignore=shutil.ignore_patterns("assets", "*.egg-info", "__pycache__", "*.pyc"),
        symlinks=True,
    )


def build_docker_image() -> None:
    stage_robotic_grounding()
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "-f", dockerfile_path, root_dir], check=True)


if __name__ == "__main__":
    build_docker_image()
