# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Export per-frame multiview fused point clouds as colored PLY files."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from v2d.mv.rig import RigConfig

from v2d.mv.postprocess.lib.fuse_depth_pointcloud import (
    fuse_multiview_depth,
    save_colored_ply,
)

logger = logging.getLogger(__name__)


def export_fused_pointcloud(
    depth_dirs: list[Path],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    cam_resolutions: list[np.ndarray],
    rgb_dirs: list[Path],
    output_dir: Path,
    frame_indices: list[int],
    mask_dirs: list[Path] | None = None,
    max_depth: float = 5.0,
    image_scale: float = 0.5,
    voxel_size: float = 0.005,
    statistical_outlier_neighbors: int = 20,
    statistical_outlier_std: float = 2.0,
) -> None:
    """Fuse depth from multiple cameras and save per-frame colored PLY files.

    Each frame is fused independently and written to
    ``{output_dir}/{frame_stem}.ply`` where ``frame_stem`` matches the source
    depth image filename (e.g. ``000000.ply``).
    """
    from v2d.common.video import FrameSource

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_source = FrameSource.from_path(depth_dirs[0])

    for fi in frame_indices:
        if fi >= ref_source.n_frames:
            logger.warning("Frame index %d out of range (max %d), skipping", fi, ref_source.n_frames - 1)
            continue

        frame_stem = ref_source.stems[fi]

        logger.info("Fusing frame %d (%s) …", fi, frame_stem)
        points, colors = fuse_multiview_depth(
            frame_indices=[fi],
            depth_dirs=depth_dirs,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            cam_resolutions=cam_resolutions,
            rgb_dirs=rgb_dirs,
            mask_dirs=mask_dirs,
            max_depth=max_depth,
            image_scale=image_scale,
            voxel_size=voxel_size,
            statistical_outlier_neighbors=statistical_outlier_neighbors,
            statistical_outlier_std=statistical_outlier_std,
        )

        ply_path = output_dir / f"{frame_stem}.ply"
        save_colored_ply(ply_path, points, colors)
        logger.info("  -> %s  (%d points)", ply_path, len(points))


def export_fused_pointcloud_from_config(cfg) -> None:
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    cam_intrinsics: list[np.ndarray] = []
    cam_extrinsics: list[np.ndarray] = []
    cam_resolutions: list[np.ndarray] = []
    depth_dirs: list[Path] = []
    rgb_dirs: list[Path] = []
    mask_dirs: list[Path] | None = None
    if cfg.get("mask_dir"):
        mask_dirs = []

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_intrinsics.append(cam.param.K)
        cam_extrinsics.append(cam.param.T)
        cam_resolutions.append(cam.param.resolution)
        depth_dirs.append(Path(cfg.depth_path_template.format(cam_name=cam.name)))
        rgb_dirs.append(Path(cfg.rgb_path_template.format(cam_name=cam.name)))
        if mask_dirs is not None:
            mask_dirs.append(
                Path(cfg.mask_path_template.format(
                    cam_name=cam.name, mask_obj_id=cfg.mask_obj_id
                ))
            )

    frame_indices = list(range(cfg.frame_start, cfg.frame_stop, cfg.frame_step))

    export_fused_pointcloud(
        depth_dirs=depth_dirs,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        cam_resolutions=cam_resolutions,
        rgb_dirs=rgb_dirs,
        output_dir=Path(cfg.output_path),
        frame_indices=frame_indices,
        mask_dirs=mask_dirs,
        max_depth=cfg.max_depth,
        image_scale=cfg.image_scale,
        voxel_size=cfg.voxel_size,
        statistical_outlier_neighbors=cfg.statistical_outlier_neighbors,
        statistical_outlier_std=cfg.statistical_outlier_std,
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export fused multiview point clouds as PLY")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--rgb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_export_fused_pointcloud.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides = {
        "camera_params_path": args.camera_params_path,
        "depth_dir": args.depth_dir,
        "rgb_dir": args.rgb_dir,
        "output_dir": args.output_dir,
    }
    if args.mask_dir is not None:
        overrides["mask_dir"] = args.mask_dir
    cfg = OmegaConf.merge(cfg, overrides)
    export_fused_pointcloud_from_config(cfg)
