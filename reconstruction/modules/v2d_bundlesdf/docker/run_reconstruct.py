"""
Run BundleSDF SDF training and texture baking with pre-computed depth and masks.

Expects the output directory to already contain:
  - keyframes.yml   (pre-computed camera poses)
  - left/           (RGB images)
  - depth/          (depth maps — one per keyframe)
  - masks/          (object masks — one per keyframe)

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
) -> None:
    inputs = {"weights_dir": weights_dir}
    if config:
        inputs["config"] = config

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
    parser.add_argument("--output_path",     required=True, help="Directory containing keyframes.yml, left/, depth/, masks/")
    parser.add_argument("--weights_dir",     required=True, help="Root weights directory (roma/ subdirs)")
    parser.add_argument("--config",          default=None,  help="NeRF config YAML path (host-side)")
    parser.add_argument("--bbox_str",        default=None,  help="Bounding box 'x1,y1,x2,y2' (informational only)")
    parser.add_argument("--skip-texture",    action="store_true", help="Skip texture baking")
    parser.add_argument("--skip-sdf",        action="store_true", help="Skip SDF training; reuse existing model_latest.pth")
    parser.add_argument("--gpu_id",          type=int, default=None)
    parser.add_argument("--dev",             action="store_true", help="Mount local modules for development")
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
    )
