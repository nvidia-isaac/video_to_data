# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_poses(
    video_path: str,
    depth_folder: str,
    masks_folder: str,
    camera_intrinsics_path: str,
    mesh_path: str,
    poses_dir: str,
    weights_dir: str,
    reference_frame: int = 0,
    target_width: int = None,
    target_height: int = None,
    reregister_iou_thresh: float = None,
    register_iteration: int = 10,
    track_iteration: int = 5,
    n_particles: int = 1,
    particle_process_noise_t: float = 0.005,
    particle_process_noise_r: float = 0.02,
    particle_iteration: int = 3,
    particle_mask_iou_weight: float = 1.0,
    mask_depth: bool = False,
    dev: bool = False,
) -> None:
    weights_abs = os.path.abspath(weights_dir)
    weights_container = f"/data/weights_dir/{os.path.basename(weights_abs)}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_video_to_poses",
        inputs={"video_path": video_path, "depth_folder": depth_folder, "masks_folder": masks_folder, "camera_intrinsics_path": camera_intrinsics_path, "mesh_path": mesh_path, "weights_dir": weights_dir},
        outputs={"poses_dir": poses_dir},
        extra_args={
            "reference_frame":          reference_frame,
            "target_width":             target_width,
            "target_height":            target_height,
            "reregister_iou_thresh":    reregister_iou_thresh,
            "register_iteration":       register_iteration,
            "track_iteration":          track_iteration,
            "n_particles":               n_particles,
            "particle_process_noise_t":  particle_process_noise_t,
            "particle_process_noise_r":  particle_process_noise_r,
            "particle_iteration":        particle_iteration,
            "particle_mask_iou_weight":  particle_mask_iou_weight,
            "mask_depth":                mask_depth,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"FOUNDATIONPOSE_WEIGHTS_DIR": weights_container},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FoundationPose video to poses in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--masks_folder", required=True)
    parser.add_argument("--camera_intrinsics_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--poses_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--reference_frame", type=int, default=0)
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--reregister_iou_thresh", type=float, default=None)
    parser.add_argument("--register_iteration", type=int, default=10)
    parser.add_argument("--track_iteration", type=int, default=5)
    parser.add_argument("--n_particles", type=int, default=1)
    parser.add_argument("--particle_process_noise_t", type=float, default=0.005)
    parser.add_argument("--particle_process_noise_r", type=float, default=0.02)
    parser.add_argument("--particle_iteration", type=int, default=3)
    parser.add_argument("--particle_mask_iou_weight", type=float, default=1.0)
    parser.add_argument("--mask_depth", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_video_to_poses(
        args.video_path, args.depth_folder, args.masks_folder,
        args.camera_intrinsics_path, args.mesh_path, args.poses_dir,
        args.weights_dir, reference_frame=args.reference_frame,
        target_width=args.target_width, target_height=args.target_height,
        reregister_iou_thresh=args.reregister_iou_thresh,
        register_iteration=args.register_iteration,
        track_iteration=args.track_iteration,
        n_particles=args.n_particles,
        particle_process_noise_t=args.particle_process_noise_t,
        particle_process_noise_r=args.particle_process_noise_r,
        particle_iteration=args.particle_iteration,
        particle_mask_iou_weight=args.particle_mask_iou_weight,
        mask_depth=args.mask_depth,
        dev=args.dev,
    )
