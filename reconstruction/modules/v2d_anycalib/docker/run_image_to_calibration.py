# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from v2d.docker.container import run_in_container
from v2d.anycalib.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_to_calibration(
    image_path: str,
    intrinsics_path: str,
    distortion_path: str,
    weights_path: str,
    cam_id: str = "kb:4",
    model_id: str = "anycalib_gen",
    undistorted_image_path: str | None = None,
    undistorted_intrinsics_path: str | None = None,
    balance: float = 0.0,
    dev: bool = False,
) -> None:
    outputs = {"intrinsics_path": intrinsics_path, "distortion_path": distortion_path}
    if undistorted_image_path is not None:
        outputs["undistorted_image_path"] = undistorted_image_path
    if undistorted_intrinsics_path is not None:
        outputs["undistorted_intrinsics_path"] = undistorted_intrinsics_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.anycalib.lib.image_to_calibration",
        inputs={"image_path": image_path, "weights_path": weights_path},
        outputs=outputs,
        extra_args={"cam_id": cam_id, "model_id": model_id, "balance": balance},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate camera calibration from a single image")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--distortion_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--cam_id", type=str, default="kb:4")
    parser.add_argument("--model_id", type=str, default="anycalib_gen")
    parser.add_argument("--undistorted_image_path", type=str, default=None)
    parser.add_argument("--undistorted_intrinsics_path", type=str, default=None)
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_image_to_calibration(
        image_path=args.image_path,
        intrinsics_path=args.intrinsics_path,
        distortion_path=args.distortion_path,
        weights_path=args.weights_path,
        cam_id=args.cam_id,
        model_id=args.model_id,
        undistorted_image_path=args.undistorted_image_path,
        undistorted_intrinsics_path=args.undistorted_intrinsics_path,
        balance=args.balance,
        dev=args.dev,
    )
