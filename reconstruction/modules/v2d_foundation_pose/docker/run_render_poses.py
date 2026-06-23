# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_poses(
    mesh_path: str,
    poses_dir: str,
    frames_dir: str,
    intrinsics_path: str,
    output_dir: str,
    batch_size: int = 32,
    use_light: bool = True,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_render_poses",
        inputs={
            "mesh_path":       mesh_path,
            "poses_dir":       poses_dir,
            "frames_dir":      frames_dir,
            "intrinsics_path": intrinsics_path,
        },
        outputs={"output_dir": output_dir},
        extra_args={
            "batch_size": batch_size,
            "use_light":  use_light,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU-batched mesh overlay renderer in Docker")
    parser.add_argument("--mesh_path",       required=True)
    parser.add_argument("--poses_dir",       required=True)
    parser.add_argument("--frames_dir",      required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--batch_size",      type=int,  default=32)
    parser.add_argument("--use_light",       action="store_true", default=True)
    parser.add_argument("--no_light",        dest="use_light", action="store_false")
    parser.add_argument("--dev",             action="store_true")
    args = parser.parse_args()
    run_render_poses(
        args.mesh_path,
        args.poses_dir,
        args.frames_dir,
        args.intrinsics_path,
        args.output_dir,
        batch_size=args.batch_size,
        use_light=args.use_light,
        dev=args.dev,
    )
