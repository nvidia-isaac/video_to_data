# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os

from v2d.cari4d.docker._config import IMAGE_NAME, MODULES_DIR
from v2d.docker.container import run_in_container


_DEV_PRESERVE_VOLUMES = ["/workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build", "/workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda"]


def run_inference(video_path: str, mask_h5_path: str, object_mesh_path: str, weights_path: str, output_dir: str, *, download_models: bool = True, expected_frames: int | None = None, device: str = "cuda", moge_devices: str | None = None, moge_batch_size: int = 8, sam3d_chunk_size: int = 16, sam3d_alignment_workers: int = 1, sam3d_refit_batch_size: int = 512, depth_alignment_render_batch_size: int = 4, depth_alignment_encoding_workers: int = 16, depth_alignment_write_batch_size: int = 64, foundationpose_iteration: int = 5, foundationpose_max_attempts: int = 5, coconet_stride: int = 96, coconet_render_batch_size: int = 32, coconet_crop_workers: int = 8, coconet_crop_buffer_count: int = 2, postopt_num_steps: int = 300, postopt_batch_size: int = 0, postopt_temporal_weight: float = 100.0, postopt_human_pose_prior_weight: float = 200.0, postopt_contact_activation_distance_m: float = 0.05, postopt_report_every: int = 10, postopt_diagnostics_every: int = 500, symmetric_object: bool = False, optimize_object_rotation: bool = False, render_batch_size: int = 4, overwrite: bool = False, dev: bool = False) -> None:
    extra_args = {"skip_weight_download": not download_models, "expected_frames": expected_frames, "device": device, "moge_devices": moge_devices, "moge_batch_size": moge_batch_size, "sam3d_chunk_size": sam3d_chunk_size, "sam3d_alignment_workers": sam3d_alignment_workers, "sam3d_refit_batch_size": sam3d_refit_batch_size, "depth_alignment_render_batch_size": depth_alignment_render_batch_size, "depth_alignment_encoding_workers": depth_alignment_encoding_workers, "depth_alignment_write_batch_size": depth_alignment_write_batch_size, "foundationpose_iteration": foundationpose_iteration, "foundationpose_max_attempts": foundationpose_max_attempts, "coconet_stride": coconet_stride, "coconet_render_batch_size": coconet_render_batch_size, "coconet_crop_workers": coconet_crop_workers, "coconet_crop_buffer_count": coconet_crop_buffer_count, "postopt_num_steps": postopt_num_steps, "postopt_batch_size": postopt_batch_size, "postopt_temporal_weight": postopt_temporal_weight, "postopt_human_pose_prior_weight": postopt_human_pose_prior_weight, "postopt_contact_activation_distance_m": postopt_contact_activation_distance_m, "postopt_report_every": postopt_report_every, "postopt_diagnostics_every": postopt_diagnostics_every, "symmetric_object": symmetric_object, "optimize_object_rotation": optimize_object_rotation, "render_batch_size": render_batch_size, "overwrite": overwrite}
    env = {"PYTHONUNBUFFERED": "1"}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    run_in_container(image=IMAGE_NAME, module="v2d.cari4d.lib.run_inference", inputs={"video_path": video_path, "mask_h5_path": mask_h5_path, "object_mesh_path": object_mesh_path}, outputs={"weights_path": weights_path, "output_dir": output_dir}, extra_args=extra_args, dev=dev, modules_dir=MODULES_DIR, gpus=True, env=env, extra_volumes=_DEV_PRESERVE_VOLUMES if dev else None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run complete CARI4D wild-video inference in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--mask_h5_path", required=True)
    parser.add_argument("--object_mesh_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_weight_download", action="store_true")
    parser.add_argument("--expected_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moge_devices", default=None)
    parser.add_argument("--moge_batch_size", type=int, default=8)
    parser.add_argument("--sam3d_chunk_size", type=int, default=16)
    parser.add_argument("--sam3d_alignment_workers", type=int, default=1)
    parser.add_argument("--sam3d_refit_batch_size", type=int, default=512)
    parser.add_argument("--depth_alignment_render_batch_size", type=int, default=4)
    parser.add_argument("--depth_alignment_encoding_workers", type=int, default=16)
    parser.add_argument("--depth_alignment_write_batch_size", type=int, default=64)
    parser.add_argument("--foundationpose_iteration", type=int, default=5)
    parser.add_argument("--foundationpose_max_attempts", type=int, default=5)
    parser.add_argument("--coconet_stride", type=int, default=96)
    parser.add_argument("--coconet_render_batch_size", type=int, default=32)
    parser.add_argument("--coconet_crop_workers", type=int, default=8)
    parser.add_argument("--coconet_crop_buffer_count", type=int, default=2)
    parser.add_argument("--postopt_num_steps", type=int, default=300)
    parser.add_argument("--postopt_batch_size", type=int, default=0)
    parser.add_argument("--postopt_temporal_weight", type=float, default=100.0)
    parser.add_argument("--postopt_human_pose_prior_weight", type=float, default=200.0)
    parser.add_argument("--postopt_contact_activation_distance_m", type=float, default=0.05)
    parser.add_argument("--postopt_report_every", type=int, default=10)
    parser.add_argument("--postopt_diagnostics_every", type=int, default=500)
    parser.add_argument("--symmetric_object", action="store_true")
    parser.add_argument("--optimize_object_rotation", action="store_true")
    parser.add_argument("--render_batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dev", action="store_true")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    values = vars(args)
    values["download_models"] = not values.pop("skip_weight_download")
    run_inference(**values)
