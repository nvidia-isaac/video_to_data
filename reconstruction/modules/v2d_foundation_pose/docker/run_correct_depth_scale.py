# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_correct_depth_scale(
    poses_dir: str,
    mesh_path: str,
    depth_folder: str,
    intrinsics_path: str,
    output_folder: str,
    masks_folder: str = None,
    smoothing_window: int = 11,
    min_valid_pixels: int = 50,
    batch_size: int = 32,
    dev: bool = False,
) -> None:
    inputs = {
        "poses_dir":       poses_dir,
        "mesh_path":       mesh_path,
        "depth_folder":    depth_folder,
        "intrinsics_path": intrinsics_path,
    }
    if masks_folder is not None:
        inputs["masks_folder"] = masks_folder
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_correct_depth_scale",
        inputs=inputs,
        outputs={"output_folder": output_folder},
        extra_args={
            "smoothing_window": smoothing_window,
            "min_valid_pixels": min_valid_pixels,
            "batch_size":       batch_size,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FP-guided depth scale correction in Docker")
    parser.add_argument("--poses_dir",         required=True)
    parser.add_argument("--mesh_path",         required=True)
    parser.add_argument("--depth_folder",      required=True)
    parser.add_argument("--intrinsics_path",   required=True)
    parser.add_argument("--output_folder",     required=True)
    parser.add_argument("--masks_folder",      default=None)
    parser.add_argument("--smoothing_window",  type=int,  default=11)
    parser.add_argument("--min_valid_pixels",  type=int,  default=50)
    parser.add_argument("--batch_size",        type=int,  default=32)
    parser.add_argument("--dev",               action="store_true")
    args = parser.parse_args()
    run_correct_depth_scale(
        args.poses_dir,
        args.mesh_path,
        args.depth_folder,
        args.intrinsics_path,
        args.output_folder,
        masks_folder=args.masks_folder,
        smoothing_window=args.smoothing_window,
        min_valid_pixels=args.min_valid_pixels,
        batch_size=args.batch_size,
        dev=args.dev,
    )
