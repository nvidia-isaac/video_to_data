# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_estimate_mesh_scale(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    weights_dir: str,
    scale_path: str,
    rescaled_mesh_path: str = None,
    pose_path: str = None,
    lo: float = 0.5,
    hi: float = 2.0,
    n_samples: int = 7,
    n_levels: int = 3,
    iou_weight: float = 1.0,
    depth_weight: float = 1.0,
    chamfer_weight: float = 0.0,
    registration_iterations: int = 5,
    dev: bool = False,
) -> None:
    weights_abs = os.path.abspath(weights_dir)
    weights_container = f"/data/weights_dir/{os.path.basename(weights_abs)}"
    inputs = {
        "mesh_path": mesh_path,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "mask_path": mask_path,
        "intrinsics_path": intrinsics_path,
        "weights_dir": weights_dir,
    }
    outputs = {"scale_path": scale_path}
    if rescaled_mesh_path is not None:
        outputs["rescaled_mesh_path"] = rescaled_mesh_path
    if pose_path is not None:
        outputs["pose_path"] = pose_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_estimate_mesh_scale",
        inputs=inputs,
        outputs=outputs,
        extra_args={
            "lo": lo,
            "hi": hi,
            "n_samples": n_samples,
            "n_levels": n_levels,
            "iou_weight": iou_weight,
            "depth_weight": depth_weight,
            "chamfer_weight": chamfer_weight,
            "registration_iterations": registration_iterations,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={
            "FOUNDATIONPOSE_WEIGHTS_DIR": weights_container,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate mesh scale via coarse-to-fine grid search in Docker")
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--rgb_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--scale_path", required=True)
    parser.add_argument("--rescaled_mesh_path", default=None)
    parser.add_argument("--pose_path", default=None)
    parser.add_argument("--lo", type=float, default=0.5)
    parser.add_argument("--hi", type=float, default=2.0)
    parser.add_argument("--n_samples", type=int, default=7)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--iou_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--chamfer_weight", type=float, default=0.0)
    parser.add_argument("--registration_iterations", type=int, default=5)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_estimate_mesh_scale(
        args.mesh_path,
        args.rgb_path,
        args.depth_path,
        args.mask_path,
        args.intrinsics_path,
        args.weights_dir,
        args.scale_path,
        rescaled_mesh_path=args.rescaled_mesh_path,
        pose_path=args.pose_path,
        lo=args.lo,
        hi=args.hi,
        n_samples=args.n_samples,
        n_levels=args.n_levels,
        iou_weight=args.iou_weight,
        depth_weight=args.depth_weight,
        chamfer_weight=args.chamfer_weight,
        registration_iterations=args.registration_iterations,
        dev=args.dev,
    )
