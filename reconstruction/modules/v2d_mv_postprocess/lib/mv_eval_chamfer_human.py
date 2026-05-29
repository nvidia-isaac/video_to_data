# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from v2d.common.datatypes import CameraIntrinsics
from v2d.mv.rig import RigConfig

from v2d.mv.postprocess.lib.mv_eval_chamfer import mv_eval_chamfer


def mv_eval_chamfer_human_from_config(cfg):
    """Load human mesh data and run mesh-agnostic chamfer evaluation."""
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    mhr_mesh = torch.load(cfg.mhr_mesh_mv_path, weights_only=False)
    mesh_verts = mhr_mesh["pred_vertices"].cpu().numpy()
    faces = mhr_mesh["faces"].cpu().numpy()

    cam_names: list[str] = []
    cam_intrinsics: list[np.ndarray] = []
    cam_extrinsics: list[np.ndarray] = []
    depth_dirs: list[Path] = []
    mask_dirs: list[Path] = []

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_names.append(cam.name)
        cam_extrinsics.append(cam.param.T)

        intrinsics_dir = Path(cfg.depth_intrinsics_path_template.format(cam_name=cam.name))
        first_json = next(intrinsics_dir.glob("*.json"), None)
        if first_json is None:
            raise FileNotFoundError(f"No intrinsics JSON found in {intrinsics_dir}")
        K = CameraIntrinsics.load(str(first_json)).to_matrix()
        cam_intrinsics.append(K)

        depth_dirs.append(Path(cfg.depth_path_template.format(cam_name=cam.name)))
        mask_dirs.append(Path(cfg.mask_path_template.format(cam_name=cam.name)))

    output_path = Path(cfg.output_path)
    eval_image_size = tuple(cfg.eval_image_size) if cfg.get("eval_image_size") else None
    debug = cfg.get("debug", 0)
    vis_dir = Path(cfg.vis_path) if cfg.get("vis_path") else None

    return mv_eval_chamfer(
        cam_names=cam_names,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        depth_dirs=depth_dirs,
        mask_dirs=mask_dirs,
        faces=faces,
        mesh_verts=mesh_verts,
        output_path=output_path,
        eval_image_size=eval_image_size,
        anomaly_median_mm=float(cfg.get("anomaly_median_mm", 30.0)),
        anomaly_outlier_pct=float(cfg.get("anomaly_outlier_pct", 10.0)),
        debug=debug,
        vis_dir=vis_dir,
        tile_shape=tuple(cfg.get("tile_shape", [2, 2])),
        tile_image_size=tuple(cfg.get("tile_image_size", [768, 576])),
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(
        description="Chamfer distance evaluation for human mesh"
    )
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--human_pose_dir", type=str, required=True,
                        help="Directory containing mhr_mesh_mv.pt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for metrics JSON and heatmap videos")
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_eval_chamfer_human.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides = {
        "camera_params_path": args.camera_params_path,
        "human_pose_dir": args.human_pose_dir,
        "output_dir": args.output_dir,
        "depth_dir": args.depth_dir,
        "mask_dir": args.mask_dir,
    }
    cfg = OmegaConf.merge(cfg, overrides)
    mv_eval_chamfer_human_from_config(cfg)
