# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Multi-view stereo depth estimation using Foundation Stereo TRT.

Iterates over stereo pairs from a rig config, reads calibration from EDEX,
and runs TRT inference per pair. Supports optional scale for output resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from v2d.mv.rig import RigConfig, CameraParam

from .image_list_to_depth import image_list_to_depth


def baseline_from_stereo_pair(left_param: CameraParam, right_param: CameraParam) -> float:
    """Extract stereo baseline (meters) from rectified camera parameters.

    For rectified pairs, the right camera's projection matrix P encodes:
        P_right[0, 3] = -fx * baseline
    """
    if right_param.P is None:
        raise ValueError("Right camera has no projection matrix P; cannot extract baseline")
    fx = right_param.P[0, 0]
    tx = right_param.P[0, 3]
    return abs(tx) / fx


def mv_image_list_to_depth_from_config(cfg):
    """Run stereo depth estimation for each stereo pair in the rig."""
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    scale = cfg.get("scale", 1.0)

    for pair in rig.get_stereo_pairs():
        print(f"\n=== Stereo pair: {pair.name} ===")

        left_param = pair.left.param
        right_param = pair.right.param
        if left_param is None or right_param is None:
            raise ValueError(
                f"Missing camera params for pair '{pair.name}'; "
                "ensure camera_params_path points to a valid EDEX"
            )

        baseline = baseline_from_stereo_pair(left_param, right_param)
        print(f"  Baseline: {baseline:.4f} m")

        if scale != 1.0:
            left_param = left_param.scale(scale)
            right_param = right_param.scale(scale)
            print(f"  Scale: {scale} -> resolution {left_param.resolution}")

        fx = float(left_param.K[0, 0])
        fy = float(left_param.K[1, 1])
        cx = float(left_param.K[0, 2])
        cy = float(left_param.K[1, 2])

        calibration = {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline,
        }

        left_dir = cfg.rgb_path_template.format(cam_name=pair.left.name)
        right_dir = cfg.rgb_path_template.format(cam_name=pair.right.name)

        depth_folder = cfg.depth_path_template.format(cam_name=pair.left.name)
        intrinsics_folder = cfg.intrinsics_path_template.format(cam_name=pair.left.name)

        print(f"  Left:  {left_dir}")
        print(f"  Right: {right_dir}")
        print(f"  Depth: {depth_folder}")
        print(f"  Calibration: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} b={baseline:.4f}")

        image_list_to_depth(
            left_dir=left_dir,
            right_dir=right_dir,
            depth_folder=depth_folder,
            intrinsics_folder=intrinsics_folder,
            calibration=calibration,
            model_dir=cfg.model_dir,
            scale=scale,
        )

        print(f"  Done: {pair.name}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(
        description="Multi-view stereo depth estimation with Foundation Stereo TRT"
    )
    parser.add_argument("--camera_params_path", type=str, required=True,
                        help="Path to EDEX file with camera calibration")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Root directory containing per-camera input frames")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for depth and intrinsics")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing Foundation Stereo ONNX/engine")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor for output resolution (e.g. 0.5 for half)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_image_list_to_depth.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides: dict = {
        "camera_params_path": args.camera_params_path,
        "rgb_dir": args.rgb_dir,
        "output_dir": args.output_dir,
        "model_dir": args.model_dir,
    }
    if args.scale is not None:
        overrides["scale"] = args.scale

    cfg = OmegaConf.merge(cfg, overrides)
    mv_image_list_to_depth_from_config(cfg)
