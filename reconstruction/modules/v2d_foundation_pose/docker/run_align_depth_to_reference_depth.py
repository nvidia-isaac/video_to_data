# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_align_depth_to_reference_depth(
    depth_folder: str,
    depth_reference_path: str,
    intrinsics_path: str,
    output_folder: str,
    masks_folder: str = None,
    reference_mask_path: str = None,
    n_iterations: int = 3,
    outlier_trim_ratio: float = 0.2,
    max_points: int = 20000,
    dev: bool = False,
) -> None:
    inputs = {
        "depth_folder":         depth_folder,
        "depth_reference_path": depth_reference_path,
        "intrinsics_path":      intrinsics_path,
    }
    if masks_folder is not None:
        inputs["masks_folder"] = masks_folder
    if reference_mask_path is not None:
        inputs["reference_mask_path"] = reference_mask_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_align_depth_to_reference_depth",
        inputs=inputs,
        outputs={"output_folder": output_folder},
        extra_args={
            "n_iterations":       n_iterations,
            "outlier_trim_ratio": outlier_trim_ratio,
            "max_points":         max_points,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align depth folder to reference depth via ICP + affine in Docker")
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--depth_reference_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--masks_folder", default=None)
    parser.add_argument("--reference_mask_path", default=None)
    parser.add_argument("--n_iterations", type=int, default=3)
    parser.add_argument("--outlier_trim_ratio", type=float, default=0.2)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_depth_to_reference_depth(
        args.depth_folder, args.depth_reference_path, args.intrinsics_path,
        args.output_folder,
        masks_folder=args.masks_folder,
        reference_mask_path=args.reference_mask_path,
        n_iterations=args.n_iterations,
        outlier_trim_ratio=args.outlier_trim_ratio,
        max_points=args.max_points,
        dev=args.dev,
    )
