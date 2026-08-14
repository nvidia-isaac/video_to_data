# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.geocalib.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_calibration(
    video_path: str,
    intrinsics_path: str,
    distortion_path: str,
    gravity_path: str,
    calibration_path: str,
    weights_path: str | None = None,
    weights: str = "pinhole",
    camera_model: str = "pinhole",
    num_samples: int = 8,
    device: str | None = None,
    dev: bool = False,
) -> None:
    inputs = {"video_path": video_path}
    if weights_path is not None:
        inputs["weights_path"] = weights_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.geocalib.lib.video_to_calibration",
        inputs=inputs,
        outputs={
            "intrinsics_path": intrinsics_path,
            "distortion_path": distortion_path,
            "gravity_path": gravity_path,
            "calibration_path": calibration_path,
        },
        extra_args={
            "weights": weights,
            "camera_model": camera_model,
            "num_samples": num_samples,
            "device": device,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate GeoCalib calibration from a video")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--distortion_path", required=True)
    parser.add_argument("--gravity_path", required=True)
    parser.add_argument("--calibration_path", required=True)
    parser.add_argument("--weights_path", default=None)
    parser.add_argument("--weights", choices=("pinhole", "distorted"), default="pinhole")
    parser.add_argument("--camera_model", default="pinhole",
                        choices=("pinhole", "simple_radial", "radial", "simple_divisional"))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_video_to_calibration(
        video_path=args.video_path,
        intrinsics_path=args.intrinsics_path,
        distortion_path=args.distortion_path,
        gravity_path=args.gravity_path,
        calibration_path=args.calibration_path,
        weights_path=args.weights_path,
        weights=args.weights,
        camera_model=args.camera_model,
        num_samples=args.num_samples,
        device=args.device,
        dev=args.dev,
    )

