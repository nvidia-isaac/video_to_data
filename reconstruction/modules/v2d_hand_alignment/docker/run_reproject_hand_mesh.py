# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_reproject_hand_mesh(
    input_path: str,
    target_intrinsics_path: str,
    output_path: str,
    world_results_path: str | None = None,
    pose_path: str | None = None,
    hand_intrinsics_path: str | None = None,
    apply_world_scale: bool = False,
    smooth_poses_sigma: float = 0.0,
    hand_width: int | None = None,
    hand_height: int | None = None,
    dev: bool = False,
) -> None:
    inputs = {"input_path": input_path, "target_intrinsics_path": target_intrinsics_path}
    if world_results_path is not None:
        inputs["world_results_path"] = world_results_path
    if pose_path is not None:
        inputs["pose_path"] = pose_path
    if hand_intrinsics_path is not None:
        inputs["hand_intrinsics_path"] = hand_intrinsics_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.reproject_hand_mesh",
        inputs=inputs,
        outputs={"output_path": output_path},
        extra_args={
            "apply_world_scale": apply_world_scale,
            "smooth_poses_sigma": smooth_poses_sigma if smooth_poses_sigma > 0 else None,
            "hand_width": hand_width,
            "hand_height": hand_height,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--target_intrinsics_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--world_results_path", default=None)
    parser.add_argument("--pose_path", default=None)
    parser.add_argument("--hand_intrinsics_path", default=None)
    parser.add_argument("--apply_world_scale", action="store_true")
    parser.add_argument("--smooth_poses_sigma", type=float, default=0.0)
    parser.add_argument("--hand_width", type=int, default=None)
    parser.add_argument("--hand_height", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_reproject_hand_mesh(
        args.input_path, args.target_intrinsics_path, args.output_path,
        world_results_path=args.world_results_path,
        pose_path=args.pose_path,
        hand_intrinsics_path=args.hand_intrinsics_path,
        apply_world_scale=args.apply_world_scale,
        smooth_poses_sigma=args.smooth_poses_sigma,
        hand_width=args.hand_width,
        hand_height=args.hand_height,
        dev=args.dev,
    )
