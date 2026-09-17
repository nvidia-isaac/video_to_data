#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Isaac Sim rigid-body/collider hold-and-drop tests."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from docker_runtime import (
    CONTAINER_INPUT_DIR,
    CONTAINER_MODULE_DIR,
    CONTAINER_OUTPUT_DIR,
    IMAGE_NAME,
    build_isaac_sim_command,
)

DROP_TEST_REPORT_NAME = "drop_test_result.json"
DROP_TEST_VIDEO_GLOB = "drop_test*.mp4"
DEFAULT_LIFT_HEIGHT_RATIO = 0.25


def warn_if_standing_failed(result: dict, output_dir: str) -> None:
    if result.get("standing", False):
        return
    tilt = result.get("tilt_degrees")
    limit = result.get("standing_angle_limit_degrees")
    detail = ""
    if isinstance(tilt, (int, float)) and isinstance(limit, (int, float)):
        detail = f" (tilt {tilt:.2f} deg; limit {limit:.2f} deg)"
    output_path = Path(output_dir).expanduser().resolve()
    video_paths = sorted(output_path.glob(DROP_TEST_VIDEO_GLOB))
    if len(video_paths) == 1:
        message = f"Review the drop-test video: {video_paths[0]}"
    elif video_paths:
        message = f"Review the {len(video_paths)} drop-test videos in: {output_path}"
    else:
        message = "No video was generated; rerun without --no-video to inspect the result."
    print(f"WARNING: Standing check failed{detail}. {message}", file=sys.stderr)


def _values(values: Sequence[object]) -> list[str]:
    return [str(value) for value in values]


def _validate_drop_test_options(
    *,
    physics_hz: int,
    video_fps: int,
    pre_lift_seconds: float,
    lift_seconds: float,
    hold_seconds: float,
    lift_height: float | None,
    lift_height_ratio: float,
    max_drop_seconds: float,
    settle_seconds: float,
    post_settle_seconds: float,
    initial_clearance: float,
    fallback_mass_kg: float,
    asset_scale: float,
    collision_approximation: str,
    local_up: Sequence[float] | None,
    initial_pose: str,
    recorded_support_path: str | None,
    standing_angle: float,
    linear_speed_limit: float,
    angular_speed_limit: float,
    pose_settle_position_ratio: float,
    pose_settle_angle: float,
    ground_z: float,
    ground_tolerance: float,
    contact_force_threshold: float,
    resolution: Sequence[int],
    camera_eye: Sequence[float] | None,
    camera_target: Sequence[float] | None,
    camera_distance_scale: float,
) -> None:
    durations = (
        pre_lift_seconds,
        lift_seconds,
        hold_seconds,
        max_drop_seconds,
        settle_seconds,
        post_settle_seconds,
    )
    scalar_values = (
        *durations,
        initial_clearance,
        fallback_mass_kg,
        asset_scale,
        standing_angle,
        linear_speed_limit,
        angular_speed_limit,
        pose_settle_position_ratio,
        pose_settle_angle,
        ground_z,
        ground_tolerance,
        contact_force_threshold,
        camera_distance_scale,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("drop-test numeric options must be finite")
    if physics_hz <= 0:
        raise ValueError("physics_hz must be positive")
    if video_fps <= 0 or video_fps > physics_hz:
        raise ValueError("video_fps must be positive and no greater than physics_hz")
    if any(duration < 0 for duration in durations):
        raise ValueError("simulation durations cannot be negative")
    if lift_height is not None and (
        not math.isfinite(lift_height) or lift_height <= 0
    ):
        raise ValueError("lift_height must be positive and finite")
    if not math.isfinite(lift_height_ratio) or lift_height_ratio <= 0:
        raise ValueError("lift_height_ratio must be positive and finite")
    if initial_clearance < 0:
        raise ValueError("initial_clearance cannot be negative")
    if fallback_mass_kg <= 0 or asset_scale <= 0 or camera_distance_scale <= 0:
        raise ValueError(
            "fallback_mass_kg, asset_scale, and camera_distance_scale must be positive"
        )
    if not 0 <= standing_angle <= 180:
        raise ValueError("standing_angle must be in [0, 180]")
    if min(
        linear_speed_limit,
        angular_speed_limit,
        ground_tolerance,
        contact_force_threshold,
    ) < 0:
        raise ValueError("diagnostic thresholds cannot be negative")
    if pose_settle_position_ratio <= 0 or not 0 < pose_settle_angle <= 180:
        raise ValueError(
            "pose settle position ratio must be positive and angle must be in (0, 180]"
        )
    if collision_approximation not in {"convexHull", "convexDecomposition"}:
        raise ValueError(f"Unsupported collision approximation: {collision_approximation}")
    if initial_pose not in {
        "as-authored",
        "principal-6",
        "principal-6-support",
        "recorded-support",
    }:
        raise ValueError(f"Unsupported initial pose: {initial_pose}")
    if local_up is not None:
        if len(local_up) != 3:
            raise ValueError("local_up must have 3 values")
        if not all(math.isfinite(value) for value in local_up):
            raise ValueError("local_up must be finite")
        if math.sqrt(sum(value * value for value in local_up)) == 0:
            raise ValueError("local_up cannot be the zero vector")
    if initial_pose.startswith("principal-6") and local_up is not None:
        raise ValueError("local_up cannot be combined with a principal-6 initial pose")
    if initial_pose == "recorded-support":
        if recorded_support_path is None:
            raise ValueError(
                "recorded-support initial pose requires recorded_support_path"
            )
        if local_up is not None:
            raise ValueError(
                "local_up is read from recorded_support_path and cannot be overridden"
            )
    elif recorded_support_path is not None:
        raise ValueError(
            "recorded_support_path can only be used with recorded-support initial pose"
        )
    if len(resolution) != 2 or any(value <= 0 for value in resolution):
        raise ValueError("resolution must contain 2 positive values")
    for name, vector in (("camera_eye", camera_eye), ("camera_target", camera_target)):
        if vector is not None and (
            len(vector) != 3 or not all(math.isfinite(value) for value in vector)
        ):
            raise ValueError(f"{name} must contain 3 finite values")


def build_drop_test_command(
    asset_path: str,
    output_dir: str,
    *,
    video: bool = True,
    fail_if_not_standing: bool = True,
    physics_hz: int = 60,
    video_fps: int = 30,
    pre_lift_seconds: float = 1.0,
    lift_seconds: float = 1.0,
    hold_seconds: float = 0.5,
    lift_height: float | None = None,
    lift_height_ratio: float = DEFAULT_LIFT_HEIGHT_RATIO,
    max_drop_seconds: float = 3.0,
    settle_seconds: float = 0.5,
    post_settle_seconds: float = 1.0,
    initial_clearance: float = 0.005,
    fallback_mass_kg: float = 0.3,
    asset_scale: float = 1.0,
    collision_approximation: str = "convexHull",
    local_up: Sequence[float] | None = None,
    initial_pose: str = "principal-6",
    recorded_support_path: str | None = None,
    standing_angle: float = 15.0,
    linear_speed_limit: float = 0.01,
    angular_speed_limit: float = 0.05,
    pose_settle_position_ratio: float = 0.01,
    pose_settle_angle: float = 2.0,
    ground_z: float = 0.0,
    ground_tolerance: float = 0.02,
    contact_force_threshold: float = 1e-4,
    resolution: Sequence[int] = (1280, 720),
    camera_eye: Sequence[float] | None = None,
    camera_target: Sequence[float] | None = None,
    camera_distance_scale: float = 2.2,
    image: str = IMAGE_NAME,
    cache_dir: str | None = None,
    accept_eula: bool = False,
    dev: bool = False,
    gpu_device: str | None = None,
) -> list[str]:
    _validate_drop_test_options(
        physics_hz=physics_hz,
        video_fps=video_fps,
        pre_lift_seconds=pre_lift_seconds,
        lift_seconds=lift_seconds,
        hold_seconds=hold_seconds,
        lift_height=lift_height,
        lift_height_ratio=lift_height_ratio,
        max_drop_seconds=max_drop_seconds,
        settle_seconds=settle_seconds,
        post_settle_seconds=post_settle_seconds,
        initial_clearance=initial_clearance,
        fallback_mass_kg=fallback_mass_kg,
        asset_scale=asset_scale,
        collision_approximation=collision_approximation,
        local_up=local_up,
        initial_pose=initial_pose,
        recorded_support_path=recorded_support_path,
        standing_angle=standing_angle,
        linear_speed_limit=linear_speed_limit,
        angular_speed_limit=angular_speed_limit,
        pose_settle_position_ratio=pose_settle_position_ratio,
        pose_settle_angle=pose_settle_angle,
        ground_z=ground_z,
        ground_tolerance=ground_tolerance,
        contact_force_threshold=contact_force_threshold,
        resolution=resolution,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_distance_scale=camera_distance_scale,
    )
    privacy_consent = os.environ.get("PRIVACY_CONSENT")
    environment = (
        {"PRIVACY_CONSENT": privacy_consent}
        if privacy_consent is not None
        else None
    )
    asset, command = build_isaac_sim_command(
        asset_path,
        output_dir,
        cache_dir=cache_dir,
        accept_eula=accept_eula,
        dev=dev,
        gpu_device=gpu_device,
        environment=environment,
    )
    recorded_support = None
    if recorded_support_path is not None:
        recorded_support = Path(recorded_support_path).expanduser().resolve()
        if not recorded_support.is_file() or recorded_support.suffix.lower() != ".json":
            raise ValueError(
                f"Missing or unsupported recorded-support JSON: {recorded_support}"
            )
        command.extend(
            [
                "-v",
                f"{recorded_support.parent}:/data/recorded-support:ro",
            ]
        )

    runtime = [
        f"{CONTAINER_MODULE_DIR}/runtime/drop_test.py",
        "--asset", f"{CONTAINER_INPUT_DIR}/{asset.name}",
        "--output-dir", CONTAINER_OUTPUT_DIR,
        "--headless",
        "--physics-hz", str(physics_hz),
        "--video-fps", str(video_fps),
        "--pre-lift-seconds", str(pre_lift_seconds),
        "--lift-seconds", str(lift_seconds),
        "--hold-seconds", str(hold_seconds),
        "--lift-height-ratio", str(lift_height_ratio),
        "--max-drop-seconds", str(max_drop_seconds),
        "--settle-seconds", str(settle_seconds),
        "--post-settle-seconds", str(post_settle_seconds),
        "--initial-clearance", str(initial_clearance),
        "--fallback-mass-kg", str(fallback_mass_kg),
        "--asset-scale", str(asset_scale),
        "--collision-approximation", collision_approximation,
        "--initial-pose", initial_pose,
        "--standing-angle", str(standing_angle),
        "--linear-speed-limit", str(linear_speed_limit),
        "--angular-speed-limit", str(angular_speed_limit),
        "--pose-settle-position-ratio", str(pose_settle_position_ratio),
        "--pose-settle-angle", str(pose_settle_angle),
        "--ground-z", str(ground_z),
        "--ground-tolerance", str(ground_tolerance),
        "--contact-force-threshold", str(contact_force_threshold),
        "--resolution", *_values(resolution),
        "--camera-distance-scale", str(camera_distance_scale),
    ]
    if lift_height is not None:
        runtime.extend(["--lift-height", str(lift_height)])
    if local_up is not None:
        runtime.extend(["--local-up", *_values(local_up)])
    if recorded_support is not None:
        runtime.extend(
            [
                "--recorded-support",
                f"/data/recorded-support/{recorded_support.name}",
            ]
        )
    if camera_eye is not None:
        runtime.extend(["--camera-eye", *_values(camera_eye)])
    if camera_target is not None:
        runtime.extend(["--camera-target", *_values(camera_target)])
    runtime.append("--video" if video else "--no-video")
    runtime.append(
        "--fail-if-not-standing"
        if fail_if_not_standing
        else "--no-fail-if-not-standing"
    )

    command.extend(["--entrypoint", "/isaac-sim/python.sh", image, *runtime])
    return command


def run_drop_test(
    asset_path: str,
    output_dir: str,
    *,
    video: bool = True,
    fail_if_not_standing: bool = True,
    physics_hz: int = 60,
    video_fps: int = 30,
    pre_lift_seconds: float = 1.0,
    lift_seconds: float = 1.0,
    hold_seconds: float = 0.5,
    lift_height: float | None = None,
    lift_height_ratio: float = DEFAULT_LIFT_HEIGHT_RATIO,
    max_drop_seconds: float = 3.0,
    settle_seconds: float = 0.5,
    post_settle_seconds: float = 1.0,
    initial_clearance: float = 0.005,
    fallback_mass_kg: float = 0.3,
    asset_scale: float = 1.0,
    collision_approximation: str = "convexHull",
    local_up: Sequence[float] | None = None,
    initial_pose: str = "principal-6",
    recorded_support_path: str | None = None,
    standing_angle: float = 15.0,
    linear_speed_limit: float = 0.01,
    angular_speed_limit: float = 0.05,
    pose_settle_position_ratio: float = 0.01,
    pose_settle_angle: float = 2.0,
    ground_z: float = 0.0,
    ground_tolerance: float = 0.02,
    contact_force_threshold: float = 1e-4,
    resolution: Sequence[int] = (1280, 720),
    camera_eye: Sequence[float] | None = None,
    camera_target: Sequence[float] | None = None,
    camera_distance_scale: float = 2.2,
    image: str = IMAGE_NAME,
    cache_dir: str | None = None,
    accept_eula: bool = False,
    dev: bool = False,
    gpu_device: str | None = None,
) -> dict:
    command = build_drop_test_command(
        asset_path,
        output_dir,
        video=video,
        fail_if_not_standing=fail_if_not_standing,
        physics_hz=physics_hz,
        video_fps=video_fps,
        pre_lift_seconds=pre_lift_seconds,
        lift_seconds=lift_seconds,
        hold_seconds=hold_seconds,
        lift_height=lift_height,
        lift_height_ratio=lift_height_ratio,
        max_drop_seconds=max_drop_seconds,
        settle_seconds=settle_seconds,
        post_settle_seconds=post_settle_seconds,
        initial_clearance=initial_clearance,
        fallback_mass_kg=fallback_mass_kg,
        asset_scale=asset_scale,
        collision_approximation=collision_approximation,
        local_up=local_up,
        initial_pose=initial_pose,
        recorded_support_path=recorded_support_path,
        standing_angle=standing_angle,
        linear_speed_limit=linear_speed_limit,
        angular_speed_limit=angular_speed_limit,
        pose_settle_position_ratio=pose_settle_position_ratio,
        pose_settle_angle=pose_settle_angle,
        ground_z=ground_z,
        ground_tolerance=ground_tolerance,
        contact_force_threshold=contact_force_threshold,
        resolution=resolution,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_distance_scale=camera_distance_scale,
        image=image,
        cache_dir=cache_dir,
        accept_eula=accept_eula,
        dev=dev,
        gpu_device=gpu_device,
    )
    output_path = Path(output_dir).expanduser().resolve()
    result_path = output_path / DROP_TEST_REPORT_NAME
    result_path.unlink(missing_ok=True)
    for video_path in output_path.glob(DROP_TEST_VIDEO_GLOB):
        video_path.unlink()

    completed = subprocess.run(command, check=False)
    if not result_path.is_file():
        raise RuntimeError(
            f"Isaac Sim exited with code {completed.returncode} "
            f"without creating {result_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") == "error":
        raise RuntimeError(f"Isaac Sim workflow failed: {result.get('error', 'unknown error')}")

    def map_video_files(container_paths: Sequence[str]) -> list[str]:
        return [str(output_path / Path(path).name) for path in container_paths]

    result["container_video_files"] = result.get("video_files", [])
    result["video_files"] = map_video_files(result["container_video_files"])
    for pose_result in result.get("pose_results", []):
        pose_result["container_video_files"] = pose_result.get("video_files", [])
        pose_result["video_files"] = map_video_files(
            pose_result["container_video_files"]
        )
    result["report_file"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if fail_if_not_standing and not result.get("standing", False):
        warn_if_standing_failed(result, output_dir)
    if not result.get("passed", False):
        reasons = result.get("failure_reasons") or ["unspecified failure"]
        raise RuntimeError(f"Drop test failed: {', '.join(str(reason) for reason in reasons)}")
    completed.check_returncode()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record MP4 evidence (default: enabled)",
    )
    parser.add_argument(
        "--fail-if-not-standing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the final pose to satisfy the standing diagnostic (default: enabled); "
            "video evidence is generated by default"
        ),
    )
    parser.add_argument("--physics-hz", type=int, default=60)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--pre-lift-seconds", type=float, default=1.0)
    parser.add_argument("--lift-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--lift-height", type=float)
    parser.add_argument(
        "--lift-height-ratio",
        type=float,
        default=DEFAULT_LIFT_HEIGHT_RATIO,
        help=(
            "Lift distance as a fraction of the object's largest extent "
            f"(default: {DEFAULT_LIFT_HEIGHT_RATIO})"
        ),
    )
    parser.add_argument("--max-drop-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--post-settle-seconds", type=float, default=1.0)
    parser.add_argument("--initial-clearance", type=float, default=0.005)
    parser.add_argument(
        "--fallback-mass-kg",
        type=float,
        default=0.3,
        help="Used only when the input asset has no authored mass",
    )
    parser.add_argument("--asset-scale", type=float, default=1.0)
    parser.add_argument(
        "--collision-approximation",
        choices=("convexHull", "convexDecomposition"),
        default="convexHull",
    )
    parser.add_argument("--local-up", nargs=3, type=float)
    parser.add_argument(
        "--initial-pose",
        choices=(
            "as-authored",
            "principal-6",
            "principal-6-support",
            "recorded-support",
        ),
        default="principal-6",
        help="Initial orientation policy (default: principal-6)",
    )
    parser.add_argument(
        "--recorded-support",
        dest="recorded_support_path",
        help="Strict support JSON required by recorded-support mode",
    )
    parser.add_argument(
        "--standing-angle",
        type=float,
        default=15.0,
        help="Tilt threshold for the optional standing diagnostic (default: 15)",
    )
    parser.add_argument("--linear-speed-limit", type=float, default=0.01)
    parser.add_argument("--angular-speed-limit", type=float, default=0.05)
    parser.add_argument(
        "--pose-settle-position-ratio",
        type=float,
        default=0.01,
        help=(
            "Allow full-pose translation up to this fraction of object "
            "extent (default: 0.01; clamped to 0.001-0.01 m)"
        ),
    )
    parser.add_argument(
        "--pose-settle-angle",
        type=float,
        default=2.0,
        help="Allow up to this full-pose rotation in degrees (default: 2)",
    )
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--ground-tolerance", type=float, default=0.02)
    parser.add_argument("--contact-force-threshold", type=float, default=1e-4)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--camera-eye", nargs=3, type=float)
    parser.add_argument("--camera-target", nargs=3, type=float)
    parser.add_argument("--camera-distance-scale", type=float, default=2.2)
    parser.add_argument("--image", default=IMAGE_NAME)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--gpu-device",
        help="Expose exactly one GPU to Isaac Sim (for example: 0)",
    )
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = vars(parser.parse_args())
    asset = args.pop("asset")
    output_dir = args.pop("output_dir")
    result = run_drop_test(asset, output_dir, **args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "rigid_body_pass_count": result["rigid_body_pass_count"],
                "standing_pass_count": result["standing_pass_count"],
                "representative_pose_id": result["representative_pose_id"],
                "report_file": result["report_file"],
                "video_files": result["video_files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
