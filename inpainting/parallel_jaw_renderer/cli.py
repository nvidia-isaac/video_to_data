"""Validate or render one explicit parallel-jaw robot/target combination."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .bundle import load_robot_bundle
from .external_ik import IK_CONSTRUCTOR_KWARGS, resolve_arm_ik_source
from .gripper import map_aperture_trajectory
from .inputs import load_parallel_jaw_inputs, select_preview_frame
from .provenance import build_provenance
from .render import render_parallel_jaw_robot
from .transforms import validate_transform


def _rgb(value: str) -> tuple[int, int, int]:
    try:
        components = tuple(int(component.strip()) for component in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "RGB must be comma-separated integers"
        ) from exc
    if len(components) != 3 or any(not 0 <= item <= 255 for item in components):
        raise argparse.ArgumentTypeError("RGB must contain three values in [0,255]")
    return components


def load_world_hub_from_metadata(path: str | Path) -> np.ndarray:
    """Reuse an explicitly named completed GT Vega mount as the shared hub."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        metadata = json.loads(source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not parse T_world_hub metadata {source}: {exc}"
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("state") != "complete":
        raise ValueError("T_world_hub metadata must be a completed render sidecar")
    kinematics = metadata.get("kinematics")
    if not isinstance(kinematics, dict) or "arm_center_world" not in kinematics:
        raise ValueError(
            "T_world_hub metadata must declare kinematics.arm_center_world"
        )
    return validate_transform(
        np.asarray(kinematics["arm_center_world"], dtype=np.float64),
        name="kinematics.arm_center_world reused as T_world_hub",
    )


def _resolve_world_hub(args: argparse.Namespace) -> tuple[np.ndarray, Path | None]:
    if args.T_world_hub is not None:
        return (
            validate_transform(
                np.asarray(args.T_world_hub, dtype=np.float64).reshape(4, 4),
                name="T_world_hub",
            ),
            None,
        )
    source = args.T_world_hub_metadata.resolve()
    return load_world_hub_from_metadata(source), source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--world-to-camera", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--scene-utils-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    hub = parser.add_mutually_exclusive_group(required=True)
    hub.add_argument(
        "--T-world-hub",
        dest="T_world_hub",
        type=float,
        nargs=16,
        metavar=tuple(f"M{row}{column}" for row in range(4) for column in range(4)),
        help="Explicit row-major 4x4 world-from-hub transform.",
    )
    hub.add_argument(
        "--T-world-hub-metadata",
        dest="T_world_hub_metadata",
        type=Path,
        help=(
            "Completed GT Vega render_metadata.json whose "
            "kinematics.arm_center_world is reused as T_world_hub."
        ),
    )
    parser.add_argument("--background-rgb", type=_rgb, default=(0, 0, 0))
    parser.add_argument("--max-ik-residual-m", type=float, default=0.01)
    parser.add_argument("--ik-orientation-cost", type=float, default=0.010)
    parser.add_argument(
        "--max-orientation-residual-deg",
        type=float,
        default=20.0,
    )
    parser.add_argument("--max-joint-step-rad", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preview-frame-index",
        type=int,
        help=(
            "Render exactly one source frame for non-production visual QA. "
            "Run separately for start/middle/end so temporal joint-step gates "
            "are never applied across disjoint source frames."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate contracts/assets/provenance without importing Pink or OpenGL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = load_parallel_jaw_inputs(
        target_path=args.target,
        intrinsic_path=args.intrinsics,
        world_to_camera_path=args.world_to_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    if args.preview_frame_index is not None:
        inputs = select_preview_frame(inputs, args.preview_frame_index)
    bundle = load_robot_bundle(args.bundle, require_visual_assets=True)
    T_world_hub, world_hub_source = _resolve_world_hub(args)
    arm_ik_source = resolve_arm_ik_source(args.scene_utils_root)

    left_gripper = map_aperture_trajectory(
        inputs.target["left_aperture_m"],
        side="left",
        spec=bundle.gripper_mapping,
        render_inspection=bundle.render_inspection,
    )
    right_gripper = map_aperture_trajectory(
        inputs.target["right_aperture_m"],
        side="right",
        spec=bundle.gripper_mapping,
        render_inspection=bundle.render_inspection,
    )
    if args.dry_run:
        report = {
            "schema_version": "v2d.inpainting.parallel-jaw-render-dry-run/v1",
            "state": "validated",
            "container_image": os.environ.get("V2D_RENDER_CONTAINER_IMAGE"),
            "container_image_id": os.environ.get("V2D_RENDER_CONTAINER_IMAGE_ID"),
            "geometry": inputs.geometry.as_dict(),
            "rendered_source_frame_indices": [
                int(value) for value in inputs.target["frame_indices"].tolist()
            ],
            "tracker": inputs.tracker,
            "projection_validation": inputs.projection_report(),
            "robot_bundle": bundle.as_dict(),
            "T_world_hub": T_world_hub.tolist(),
            "T_world_robot_root": bundle.world_robot_root(T_world_hub).tolist(),
            "gripper_mapping": {
                "left": dict(left_gripper.report),
                "right": dict(right_gripper.report),
            },
            "kinematics_policy": {
                "max_position_residual_m": args.max_ik_residual_m,
                "max_orientation_residual_deg": args.max_orientation_residual_deg,
                "max_joint_step_rad": args.max_joint_step_rad,
                "ik_constructor_kwargs": {
                    **IK_CONSTRUCTOR_KWARGS,
                    "orientation_cost": args.ik_orientation_cost,
                },
            },
            "provenance": build_provenance(
                target=inputs.target_path,
                intrinsic=inputs.intrinsic_path,
                world_to_camera=inputs.world_to_camera_path,
                world_hub=world_hub_source,
                bundle=bundle,
                arm_ik_source=arm_ik_source,
                capture_mode="validation_time",
            ),
            "outputs_created": False,
            "ready_for_full_render": True,
            "deferred_until_full_render": [
                "Pink/Pinocchio solve and strict residual gates",
                "yourdfpy mimic expansion",
                "pyrender rasterization",
                "MP4 encode/decode verification",
            ],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    metadata = render_parallel_jaw_robot(
        inputs,
        bundle,
        scene_utils_root=args.scene_utils_root,
        output_dir=args.output_dir,
        T_world_hub=T_world_hub,
        world_hub_source_path=world_hub_source,
        background_rgb=args.background_rgb,
        orientation_cost=args.ik_orientation_cost,
        max_position_residual_m=args.max_ik_residual_m,
        max_orientation_residual_deg=args.max_orientation_residual_deg,
        max_joint_step_rad=args.max_joint_step_rad,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
