# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_videos_to_masks.yaml"


def run_mv_videos_to_masks(
    bbox_dir: str,
    rgb_dir: str,
    output_dir: str,
    weights_dir: str,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
    gpu: int = 0,
) -> None:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative physical GPU index")
    inputs = {
        "bbox_dir": bbox_dir,
        "rgb_dir": rgb_dir,
        "weights_dir": weights_dir,
        "config_path": config_path,
    }

    outputs = {"output_dir": output_dir}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam2.lib.mv_videos_to_masks",
        inputs=inputs,
        outputs=outputs,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpu_device=gpu,
        env={"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"},
        network_disabled=True,
        strict_io_isolation=True,
        input_directories={"bbox_dir", "rgb_dir", "weights_dir"},
        output_directories={"output_dir"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-view video/image to masks using SAM2 with detectron2 bbox prompts"
    )
    parser.add_argument("--bbox_dir", type=str, required=True,
                        help="Directory containing per-camera bbox_track .pt files")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Directory containing per-camera input frames")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-camera masks")
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Path to SAM2 weights directory")
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG),
                        help="Path to config YAML")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="Physical host GPU index")
    args = parser.parse_args()

    run_mv_videos_to_masks(
        bbox_dir=args.bbox_dir,
        rgb_dir=args.rgb_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        config_path=args.config_path,
        dev=args.dev,
        gpu=args.gpu,
    )
