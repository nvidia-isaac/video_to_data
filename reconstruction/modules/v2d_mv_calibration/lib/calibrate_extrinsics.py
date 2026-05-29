# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Extrinsic calibration: chessboard detection -> PnP initialization -> Ceres BA."""

import json
import logging
from pathlib import Path

import numpy as np
import pyceres

from v2d.mv.rig import CameraParam, RigConfig
from v2d.common.video import FrameSource

from v2d.mv.calibration.lib.chessboard import chessboard_extract_correspondences
from v2d.mv.calibration.lib.solve import (
    extrinsics_estimate_pnp,
    extrinsics_solve_ba,
    reprojection_error_stats,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calibrate_extrinsics(
    rig: RigConfig,
    rgb_paths: list[Path],
    calibration_order: list[int],
    camera_params_path: Path,
    output_camera_params_path: Path,
    board_size: tuple[int, int] = (6, 10),
    square_size: float = 0.1,
    max_iterations: int = 50,
    num_workers: int = 8,
    frames_slice: slice | None = None,
    debug: int = 0,
) -> list[CameraParam]:
    """Run extrinsic calibration on a multi-camera dataset.

    Camera parameters are read from ``rig.get_camera(cam_id).param``.
    Optimized params are saved to ``output_camera_params_path``.

    Args:
        rig: RigConfig with stereo pair definitions and loaded camera params.
        rgb_paths: List of paths to RGB frames, one per camera.
        calibration_order: Left camera IDs for pairwise PnP chain.
        camera_params_path: Source camera params file (for save merge).
        output_camera_params_path: Where to write calibrated camera params.
            A sibling ``calibration_accuracy.json`` is written next to this path
            with chessboard reprojection RMSE and per-camera breakdown.
        board_size: (width, height) inner corners of chessboard.
        square_size: Chessboard square size in meters.
        max_iterations: Maximum bundle adjustment iterations.
        num_workers: Workers for chessboard detection.
        frames_slice: Optional slice to limit frame range.
        debug: Debug level. >0: save rerun visualization; >1: save reprojected points.

    Returns:
        Optimized list of CameraParam (one per camera).
    """
    logger.info(
        f"Starting extrinsics calibration"
        f"\n\t- Calibration order: {calibration_order}"
        f"\n\t- Board size: {board_size}"
        f"\n\t- Square size: {square_size}m"
        f"\n\t- Cameras: {len(rgb_paths)}"
    )

    # Extract chessboard correspondences
    correspondences, frame_indices = chessboard_extract_correspondences(
        source_paths=[Path(p) for p in rgb_paths],
        frames_slice=frames_slice,
        board_size=board_size,
        num_workers=num_workers,
    )

    # Define target 3D points
    target_xyz = np.zeros((board_size[0] * board_size[1], 3))
    target_xyz[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    target_xyz *= square_size

    # Read camera parameters from rig
    camera_params = [rig.get_camera(i).param for i in range(len(rig.cameras))]

    # Stereo pairs from rig config
    stereo_pairs = [(p.left.cam_id, p.right.cam_id) for p in rig.get_stereo_pairs()]

    # PnP initialization
    est_camera_params, est_target_poses = extrinsics_estimate_pnp(
        correspondences=correspondences,
        target_xyz=target_xyz,
        camera_params=camera_params,
        calibration_order=calibration_order,
        stereo_pairs=stereo_pairs,
    )

    # Bundle adjustment
    if debug > 0:
        summary, camera_params_history, target_poses_history = extrinsics_solve_ba(
            correspondences=correspondences,
            target_xyz=target_xyz,
            camera_params=est_camera_params,
            init_target_poses=est_target_poses,
            max_num_iterations=max_iterations,
            return_history=True,
        )
        if summary.termination_type != pyceres.TerminationType.CONVERGENCE:
            raise RuntimeError(
                f"Bundle adjustment did not converge: {summary.FullReport()}"
            )
        opt_camera_params = camera_params_history[-1]
        opt_target_poses = target_poses_history[-1]

        # Rerun visualization of optimization history
        from v2d.mv.calibration.lib.vis import visualize_camera_and_target_poses

        vis_dir = output_camera_params_path.parent
        vis_dir.mkdir(parents=True, exist_ok=True)

        num_iters = len(camera_params_history)
        num_cams = len(camera_params)
        frustum_colors = np.zeros((num_iters, num_cams, 3), dtype=np.uint8)
        t_frac = np.linspace(0, 1, num_iters)
        for t in range(num_iters):
            frustum_colors[t, :, 0] = int((1 - t_frac[t]) * 255)
            frustum_colors[t, :, 1] = int(t_frac[t] * 255)

        visualize_camera_and_target_poses(
            output_file=vis_dir / "opt_poses.rrd",
            camera_params_seq=camera_params_history,
            target_poses_seq=target_poses_history,
            frustum_colors=frustum_colors,
        )

        if debug > 1:
            from v2d.mv.calibration.lib.vis import visualize_reprojected_points

            for cam_id in calibration_order:
                cam_entry = rig.get_camera(cam_id)
                cam_name = cam_entry.name

                fs = FrameSource.from_path(rgb_paths[cam_id], frames_slice=frames_slice)
                per_cam_features = [
                    frame[cam_id] for frame in correspondences
                ]
                visualize_reprojected_points(
                    output_dir=vis_dir / cam_name,
                    frame_source=fs,
                    target_xyz=target_xyz,
                    per_cam_features=per_cam_features,
                    est_camera_param=est_camera_params[cam_id],
                    opt_camera_param=opt_camera_params[cam_id],
                    est_target_poses=est_target_poses,
                    opt_target_poses=opt_target_poses,
                    frame_indices=frame_indices,
                )
    else:
        summary, opt_camera_params, opt_target_poses = extrinsics_solve_ba(
            correspondences=correspondences,
            target_xyz=target_xyz,
            camera_params=est_camera_params,
            init_target_poses=est_target_poses,
            max_num_iterations=max_iterations,
        )
        if summary.termination_type != pyceres.TerminationType.CONVERGENCE:
            raise RuntimeError(
                f"Bundle adjustment did not converge: {summary.FullReport()}"
            )

    camera_names = [rig.get_camera(i).name for i in range(len(rig.cameras))]
    accuracy_path = output_camera_params_path.parent / "calibration_accuracy.json"
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_report = {
        "board_size": [int(board_size[0]), int(board_size[1])],
        "square_size_m": float(square_size),
        "num_calibration_frames": len(correspondences),
        "after_pnp_initialization": reprojection_error_stats(
            correspondences,
            target_xyz,
            est_camera_params,
            est_target_poses,
            camera_names,
        ),
        "after_bundle_adjustment": reprojection_error_stats(
            correspondences,
            target_xyz,
            opt_camera_params,
            opt_target_poses,
            camera_names,
        ),
    }
    accuracy_path.write_text(json.dumps(accuracy_report, indent=2), encoding="utf-8")
    logger.info(
        "Calibration accuracy (chessboard corner reprojection) written to %s",
        accuracy_path,
    )
    ba_stats = accuracy_report["after_bundle_adjustment"]
    ba_rmse = ba_stats.get("rmse_pixels")
    if ba_rmse is not None:
        logger.info(
            "Bundle-adjustment chessboard RMSE: %.4f px "
            "(median %.4f, max %.4f over %d corners)",
            ba_rmse,
            ba_stats["median_error_pixels"],
            ba_stats["max_error_pixels"],
            ba_stats["num_corners"],
        )

    for cam_id, param in enumerate(opt_camera_params):
        rig.cameras[cam_id].param = param
    rig.save_camera_params(
        source_path=camera_params_path,
        output_path=output_camera_params_path,
    )
    logger.info(f"Calibrated extrinsics written to {output_camera_params_path}")

    return opt_camera_params


def calibrate_extrinsics_from_config(cfg):
    """Resolve config fields into explicit arguments for calibrate_extrinsics."""
    camera_params_path = Path(cfg.camera_params_path)
    rig = RigConfig(cfg.rig_name, camera_params_path=camera_params_path)

    frames_slice = slice(cfg.get("start", 0), cfg.get("stop"), cfg.get("step", 1))

    input_suffix = cfg.get("input_suffix", "")
    rgb_paths: list[Path] = []
    for cam in rig.get_all_cameras():
        rgb_paths.append(Path(str(Path(cfg.rgb_dir) / cam.image_path) + input_suffix))

    calibrate_extrinsics(
        rig=rig,
        rgb_paths=rgb_paths,
        calibration_order=list(cfg.calibration_order),
        camera_params_path=camera_params_path,
        output_camera_params_path=Path(cfg.output_camera_params_path),
        board_size=tuple(cfg.get("board_size", [6, 10])),
        square_size=cfg.get("square_size", 0.1),
        max_iterations=cfg.get("max_iterations", 50),
        num_workers=cfg.get("num_workers", 8),
        frames_slice=frames_slice,
        debug=cfg.get("debug", 0),
    )


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Extrinsic camera calibration")
    parser.add_argument("--camera_params_path", type=str, required=True,
                        help="Path to camera params file (e.g. EDEX) with intrinsics")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Directory containing per-camera image subdirectories")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "calibrate_extrinsics.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides: dict = {
        "camera_params_path": args.camera_params_path,
        "rgb_dir": args.rgb_dir,
        "output_dir": args.output_dir,
    }
    if args.start is not None:
        overrides["start"] = args.start
    if args.stop is not None:
        overrides["stop"] = args.stop
    if args.step is not None:
        overrides["step"] = args.step
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers

    cfg = OmegaConf.merge(cfg, overrides)
    calibrate_extrinsics_from_config(cfg)
