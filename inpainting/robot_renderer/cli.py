"""Validate or render offline Vega + Sharpa artifacts for one source clip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .assets import resolve_robot_assets, validate_finger_trajectories
from .backend import render_robot
from .external_ik import (
    FLANGES,
    HAND_MOUNT,
    IK_CONSTRUCTOR_KWARGS,
    resolve_external_ik_sources,
)
from .inputs import load_render_inputs
from .provenance import build_provenance
from .transforms import validate_rigid_transform


def _rgb(value: str) -> tuple[int, int, int]:
    try:
        components = tuple(int(component.strip()) for component in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB must be comma-separated integers") from exc
    if len(components) != 3 or any(component < 0 or component > 255 for component in components):
        raise argparse.ArgumentTypeError("RGB must be three values in [0,255]")
    return components


def _load_transform(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=False)
    else:
        matrix = np.loadtxt(path)
    return validate_rigid_transform(matrix, name="arm_center_world")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--world-to-camera", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--scene-utils-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--arm-center-world",
        type=Path,
        help="Optional explicit 4x4 T_world_arm-center matrix (.npy or text).",
    )
    parser.add_argument("--background-rgb", type=_rgb, default=(0, 0, 0))
    parser.add_argument("--max-ik-residual-m", type=float, default=0.01)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate trajectory/calibration/assets only; import no IK/OpenGL dependencies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = load_render_inputs(
        trajectory_path=args.trajectory,
        intrinsic_path=args.intrinsics,
        world_to_camera_path=args.world_to_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    assets = resolve_robot_assets(args.asset_root)
    validate_finger_trajectories(assets, inputs.trajectory)
    external_sources = resolve_external_ik_sources(args.scene_utils_root)
    arm_center_world = _load_transform(args.arm_center_world)

    if args.dry_run:
        report = {
            "schema_version": "v2d.inpainting.robot-render-dry-run/v1",
            "state": "validated",
            "container_image": os.environ.get("V2D_RENDER_CONTAINER_IMAGE"),
            "container_image_id": os.environ.get("V2D_RENDER_CONTAINER_IMAGE_ID"),
            "geometry": inputs.geometry.as_dict(),
            "trajectory_coordinate_frame": inputs.coordinate_frame,
            "trajectory": str(inputs.trajectory_path),
            "intrinsic": inputs.intrinsic.tolist(),
            "world_to_camera": str(inputs.world_to_camera_path),
            "projection_validation": inputs.projection_report(),
            "provenance": build_provenance(
                trajectory=inputs.trajectory_path,
                intrinsic=inputs.intrinsic_path,
                world_to_camera=inputs.world_to_camera_path,
                capture_mode="validation_time",
            ),
            "assets": assets.as_dict(),
            "external_ik": external_sources.as_dict(),
            "kinematics_policy": {
                "max_position_residual_m": args.max_ik_residual_m,
                "max_joint_step_rad": args.max_joint_step_rad,
            },
            "kinematics_configuration": {
                "flanges": list(FLANGES),
                "hand_mount": HAND_MOUNT,
                "ik_constructor_kwargs": IK_CONSTRUCTOR_KWARGS,
                "arm_center_method": (
                    "explicit_world_transform"
                    if arm_center_world is not None
                    else "external_arm_mount_opt.place_hub_from_wrists"
                ),
                "arm_center_world": (
                    arm_center_world.tolist() if arm_center_world is not None else None
                ),
            },
            "outputs_created": False,
            "ready_for_full_render": True,
            "readiness_scope": "trajectory_calibration_assets_and_source_files_only",
            "deferred_until_full_render": [
                "external Pinocchio/Pink import and mount search",
                "global arm IK solve and residual validation",
                "OpenGL mesh rasterization",
                "MP4 encode/decode verification",
            ],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    metadata = render_robot(
        inputs,
        assets,
        scene_utils_root=args.scene_utils_root,
        output_dir=args.output_dir,
        arm_center_world=arm_center_world,
        background_rgb=args.background_rgb,
        max_position_residual_m=args.max_ik_residual_m,
        max_joint_step_rad=args.max_joint_step_rad,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
