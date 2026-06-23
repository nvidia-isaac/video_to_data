# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_align_depth_to_object(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    weights_dir: str,
    output_depth_path: str,
    scale_lo: float = 0.5,
    scale_hi: float = 2.0,
    shift_lo: float = -0.5,
    shift_hi: float = 0.5,
    n_scale_samples: int = 7,
    n_shift_samples: int = 5,
    n_levels: int = 3,
    iou_weight: float = 1.0,
    depth_weight: float = 1.0,
    registration_iterations: int = 5,
    dev: bool = False,
) -> None:
    weights_abs = os.path.abspath(weights_dir)
    weights_container = f"/data/weights_dir/{os.path.basename(weights_abs)}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_align_depth_to_object",
        inputs={
            "mesh_path":       mesh_path,
            "rgb_path":        rgb_path,
            "depth_path":      depth_path,
            "mask_path":       mask_path,
            "intrinsics_path": intrinsics_path,
            "weights_dir":     weights_dir,
        },
        outputs={"output_depth_path": output_depth_path},
        extra_args={
            "scale_lo":               scale_lo,
            "scale_hi":               scale_hi,
            "shift_lo":               shift_lo,
            "shift_hi":               shift_hi,
            "n_scale_samples":        n_scale_samples,
            "n_shift_samples":        n_shift_samples,
            "n_levels":               n_levels,
            "iou_weight":             iou_weight,
            "depth_weight":           depth_weight,
            "registration_iterations": registration_iterations,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"FOUNDATIONPOSE_WEIGHTS_DIR": weights_container},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align depth to object mesh via FP affine grid search in Docker")
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--rgb_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--output_depth_path", required=True)
    parser.add_argument("--scale_lo", type=float, default=0.5)
    parser.add_argument("--scale_hi", type=float, default=2.0)
    parser.add_argument("--shift_lo", type=float, default=-0.5)
    parser.add_argument("--shift_hi", type=float, default=0.5)
    parser.add_argument("--n_scale_samples", type=int, default=7)
    parser.add_argument("--n_shift_samples", type=int, default=5)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--iou_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--registration_iterations", type=int, default=5)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_depth_to_object(
        args.mesh_path, args.rgb_path, args.depth_path, args.mask_path,
        args.intrinsics_path, args.weights_dir, args.output_depth_path,
        scale_lo=args.scale_lo, scale_hi=args.scale_hi,
        shift_lo=args.shift_lo, shift_hi=args.shift_hi,
        n_scale_samples=args.n_scale_samples, n_shift_samples=args.n_shift_samples,
        n_levels=args.n_levels, iou_weight=args.iou_weight, depth_weight=args.depth_weight,
        registration_iterations=args.registration_iterations,
        dev=args.dev,
    )
