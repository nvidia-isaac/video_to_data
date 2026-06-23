# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run BundleSDF SDF training and texture baking with pre-computed depth and masks.

By default, expects the output directory to already contain:
  - keyframes.yml   (pre-computed camera poses)
  - left/           (RGB images)
  - depth/          (depth maps — one per keyframe)
  - masks/          (object masks — one per keyframe)

Custom input paths can be supplied directly via optional arguments to avoid
relying on a specific folder structure:
  --images_dir, --depth_dir, --masks_dir, --poses_file, --intrinsics_file

Outputs:
  <output_path>/textured_mesh.obj  — final textured mesh
  <output_path>/mesh_cleaned.obj   — untextured SDF mesh
"""
import argparse
import os
from pathlib import Path

from v2d.docker.container import run_in_container

IMAGE_NAME = "v2d_bundlesdf"
_MODULES_DIR = str(Path(__file__).parents[2])  # reconstruction/modules/


def run_reconstruct(
    output_path: str,
    weights_dir: str,
    config: str = None,
    bbox_str: str = None,
    skip_texture: bool = False,
    skip_sdf: bool = False,
    gpu_id: int = None,
    dev: bool = False,
    images_dir: str = None,
    depth_dir: str = None,
    masks_dir: str = None,
    poses_file: str = None,
    intrinsics_file: str = None,
) -> None:
    inputs = {"weights_dir": weights_dir}
    if config:
        inputs["config"] = config
    if images_dir:
        inputs["images_dir"] = images_dir
    if depth_dir:
        inputs["depth_dir"] = depth_dir
    if masks_dir:
        inputs["masks_dir"] = masks_dir
    if poses_file:
        inputs["poses_file"] = poses_file
    if intrinsics_file:
        inputs["intrinsics_file"] = intrinsics_file

    extra = {}
    if bbox_str:
        extra["bbox_str"] = bbox_str
    if skip_texture:
        extra["skip-texture"] = True
    if skip_sdf:
        extra["skip-sdf"] = True

    env = {}
    if gpu_id is not None:
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id), "NVIDIA_VISIBLE_DEVICES": str(gpu_id)}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d_bundlesdf.lib.reconstruct",
        inputs=inputs,
        outputs={"output_path": output_path},
        extra_args=extra,
        env=env or None,
        dev=dev,
        modules_dir=_MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BundleSDF SDF training and texture baking")
    parser.add_argument("--output_path",      required=True, help="Output directory for mesh results")
    parser.add_argument("--weights_dir",      required=True, help="Root weights directory (roma/ subdirs)")
    parser.add_argument("--config",           default=None,  help="NeRF config YAML path (host-side)")
    parser.add_argument("--bbox_str",         default=None,  help="Bounding box 'x1,y1,x2,y2' (informational only)")
    parser.add_argument("--skip-texture",     action="store_true", help="Skip texture baking")
    parser.add_argument("--skip-sdf",         action="store_true", help="Skip SDF training; reuse existing model_latest.pth")
    parser.add_argument("--gpu_id",           type=int, default=None)
    parser.add_argument("--dev",              action="store_true", help="Mount local modules for development")
    parser.add_argument("--images_dir",       default=None, help="RGB images directory (default: <output_path>/left/)")
    parser.add_argument("--depth_dir",        default=None, help="Depth maps directory (default: <output_path>/depth/)")
    parser.add_argument("--masks_dir",        default=None, help="Object masks directory (default: <output_path>/masks/)")
    parser.add_argument("--poses_file",       default=None, help="Camera poses YAML file (default: <output_path>/keyframes.yml)")
    parser.add_argument("--intrinsics_file",  default=None, help="Camera intrinsics JSON file (default: <output_path>/calibration.json)")
    args = parser.parse_args()
    run_reconstruct(
        output_path=args.output_path,
        weights_dir=args.weights_dir,
        config=args.config,
        bbox_str=args.bbox_str,
        skip_texture=args.skip_texture,
        skip_sdf=args.skip_sdf,
        gpu_id=args.gpu_id,
        dev=args.dev,
        images_dir=args.images_dir,
        depth_dir=args.depth_dir,
        masks_dir=args.masks_dir,
        poses_file=args.poses_file,
        intrinsics_file=args.intrinsics_file,
    )
