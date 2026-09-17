# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extrinsic calibration: chessboard detection -> PnP initialization -> Ceres BA."""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pyceres

from v2d.mv.rig import CameraParam, RigConfig, apply_focal_correction
from v2d.common.video import FrameSource

from v2d.mv.calibration.lib.chessboard import chessboard_extract_correspondences
from v2d.mv.calibration.lib.solve import (
    extrinsics_estimate_pnp,
    extrinsics_solve_ba,
    reprojection_error_stats,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "calibrate_extrinsics.yaml"
_CALIBRATION_SETUP_DIR = Path(__file__).parent / "calibration_setups"
_CALIBRATION_SETUP_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _load_calibration_setup(setup_name: str):
    """Load a packaged calibration setup by safe identifier."""
    from omegaconf import OmegaConf

    if not isinstance(setup_name, str) or not _CALIBRATION_SETUP_PATTERN.fullmatch(
        setup_name
    ):
        raise ValueError(
            "Invalid calibration setup identifier "
            f"{setup_name!r}; use only letters, numbers, underscores, and hyphens"
        )

    setup_path = _CALIBRATION_SETUP_DIR / f"{setup_name}.yaml"
    if not setup_path.is_file():
        raise ValueError(f"Unknown calibration setup: {setup_name!r}")
    return OmegaConf.load(setup_path)


def load_calibration_config(
    config_path: str | Path | None = None,
    calibration_setup: str | None = None,
    overrides: dict | None = None,
):
    """Load defaults, an optional packaged setup, and per-run overrides.

    Merge precedence is module defaults, selected setup, override config, then
    explicit overrides. An explicit ``calibration_setup`` takes precedence over
    a selector in the override config.
    """
    from omegaconf import OmegaConf

    default_cfg = OmegaConf.load(_DEFAULT_CONFIG_PATH)
    override_cfg = (
        OmegaConf.load(config_path) if config_path is not None else OmegaConf.create()
    )

    setup_name = calibration_setup
    if setup_name is None:
        setup_name = override_cfg.get(
            "calibration_setup",
            default_cfg.get("calibration_setup"),
        )

    setup_cfg = (
        _load_calibration_setup(setup_name)
        if setup_name is not None
        else OmegaConf.create()
    )
    explicit_overrides = dict(overrides or {})
    if calibration_setup is not None:
        explicit_overrides["calibration_setup"] = calibration_setup

    cfg = OmegaConf.merge(
        default_cfg,
        setup_cfg,
        override_cfg,
        explicit_overrides,
    )
    logger.info(
        "Resolved calibration configuration"
        "\n\t- Setup: %s"
        "\n\t- Rig: %s"
        "\n\t- Calibration order: %s"
        "\n\t- Board size: %s"
        "\n\t- Square size: %sm"
        "\n\t- Focal corrections: %s"
        "\n\t- Chessboard detector: %s",
        cfg.get("calibration_setup") or "legacy defaults",
        cfg.rig_name,
        list(cfg.calibration_order),
        tuple(cfg.board_size),
        cfg.square_size,
        dict(cfg.get("correction_focal", {})),
        "marker_sb" if cfg.get("use_marker_chessboard", False) else "legacy",
    )
    return cfg


def _apply_focal_corrections(
    rig: RigConfig,
    correction_focal: dict[int, float] | None,
) -> dict[int, float]:
    """Validate and apply per-camera focal correction factors to a rig."""
    corrections = {
        int(cam_id): float(factor)
        for cam_id, factor in (correction_focal or {}).items()
    }

    for cam_id, factor in corrections.items():
        if cam_id not in rig.cameras:
            raise ValueError(f"Focal correction references unknown camera ID {cam_id}")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"Focal correction for camera {cam_id} must be finite and positive: "
                f"{factor}"
            )
        if rig.get_camera(cam_id).param is None:
            raise ValueError(f"Camera {cam_id} has no parameters for focal correction")

    for cam_id, factor in corrections.items():
        logger.warning("Applying focal correction %s to camera %d", factor, cam_id)
        apply_focal_correction(rig.get_camera(cam_id).param, factor)

    return corrections


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
    use_marker_chessboard: bool = False,
    correction_focal: dict[int, float] | None = None,
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
        use_marker_chessboard: Use OpenCV's marker-aware SB detector for a
            three-dot asymmetric checkerboard. Defaults to the legacy detector.
        correction_focal: Optional focal correction factor keyed by camera ID.
            Corrections are applied before PnP and bundle adjustment and are
            persisted in the output camera parameters.

    Returns:
        Optimized list of CameraParam (one per camera).
    """
    correction_focal = _apply_focal_corrections(rig, correction_focal)

    logger.info(
        f"Starting extrinsics calibration"
        f"\n\t- Calibration order: {calibration_order}"
        f"\n\t- Board size: {board_size}"
        f"\n\t- Square size: {square_size}m"
        f"\n\t- Chessboard detector: "
        f"{'marker_sb' if use_marker_chessboard else 'legacy'}"
        f"\n\t- Focal corrections: {correction_focal}"
        f"\n\t- Cameras: {len(rgb_paths)}"
    )

    # Extract chessboard correspondences
    correspondences, frame_indices = chessboard_extract_correspondences(
        source_paths=[Path(p) for p in rgb_paths],
        frames_slice=frames_slice,
        board_size=board_size,
        num_workers=num_workers,
        use_marker_chessboard=use_marker_chessboard,
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
        "correction_focal": {
            str(cam_id): factor
            for cam_id, factor in sorted(correction_focal.items())
        },
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
    correction_focal_raw = cfg.get("correction_focal", {})
    correction_focal = {
        int(cam_id): float(factor)
        for cam_id, factor in correction_focal_raw.items()
    }
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
        use_marker_chessboard=cfg.get("use_marker_chessboard", False),
        correction_focal=correction_focal,
    )


def _main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extrinsic camera calibration")
    parser.add_argument("--camera_params_path", type=str, required=True,
                        help="Path to camera params file (e.g. EDEX) with intrinsics")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Directory containing per-camera image subdirectories")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    parser.add_argument(
        "--calibration_setup",
        type=str,
        default=None,
        help="Packaged calibration setup identifier (without .yaml)",
    )
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--use_marker_chessboard",
        action="store_true",
        help="Use marker-aware SB detection for a three-dot asymmetric checkerboard",
    )
    args = parser.parse_args(argv)

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
    if args.use_marker_chessboard:
        overrides["use_marker_chessboard"] = True

    cfg = load_calibration_config(
        config_path=args.config_path,
        calibration_setup=args.calibration_setup,
        overrides=overrides,
    )
    calibrate_extrinsics_from_config(cfg)


if __name__ == "__main__":
    _main()
