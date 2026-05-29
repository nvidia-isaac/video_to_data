# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Docker wrapper for single stereo pair preprocessing."""

from v2d.docker.container import run_in_container
from v2d.mv.preprocess.docker._config import IMAGE_NAME, MODULES_DIR


def run_preprocess_stereo(
    left_image_dir: str,
    right_image_dir: str,
    left_output_image_dir: str,
    right_output_image_dir: str,
    camera_params_path: str,
    left_cam_id: int,
    right_cam_id: int,
    scale: float = 1.0,
    output_resolution: tuple[int, int] | None = None,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    num_workers: int | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "left_image_dir": left_image_dir,
        "right_image_dir": right_image_dir,
        "camera_params_path": camera_params_path,
    }

    outputs = {
        "left_output_image_dir": left_output_image_dir,
        "right_output_image_dir": right_output_image_dir,
    }

    extra_args = {
        "left_cam_id": left_cam_id,
        "right_cam_id": right_cam_id,
        "scale": scale,
        "start": start,
        "step": step,
    }
    if output_resolution is not None:
        extra_args["output_resolution"] = list(output_resolution)
    if stop is not None:
        extra_args["stop"] = stop
    if num_workers is not None:
        extra_args["num_workers"] = num_workers

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.preprocess.lib.preprocess_stereo",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single stereo pair preprocessing in Docker")
    parser.add_argument("--left_image_dir", type=str, required=True)
    parser.add_argument("--right_image_dir", type=str, required=True)
    parser.add_argument("--left_output_image_dir", type=str, required=True)
    parser.add_argument("--right_output_image_dir", type=str, required=True)
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--left_cam_id", type=int, required=True)
    parser.add_argument("--right_cam_id", type=int, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output_resolution", type=int, nargs=2, default=None,
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_preprocess_stereo(
        left_image_dir=args.left_image_dir,
        right_image_dir=args.right_image_dir,
        left_output_image_dir=args.left_output_image_dir,
        right_output_image_dir=args.right_output_image_dir,
        camera_params_path=args.camera_params_path,
        left_cam_id=args.left_cam_id,
        right_cam_id=args.right_cam_id,
        scale=args.scale,
        output_resolution=tuple(args.output_resolution) if args.output_resolution else None,
        start=args.start,
        stop=args.stop,
        step=args.step,
        num_workers=args.num_workers,
        dev=args.dev,
    )
