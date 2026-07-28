"""Shared-hub target conversion and strict dual-arm Pink IK validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bundle import RobotBundle
from .external_ik import IK_CONSTRUCTOR_KWARGS, load_arm_ik
from .inputs import ParallelJawInputs
from .transforms import (
    invert_transform,
    matrix_to_quaternion_wxyz,
    orientation_error_degrees,
    validate_transform,
)


class KinematicsError(RuntimeError):
    """Raised when IK output fails an explicit quality or joint gate."""


@dataclass(frozen=True)
class KinematicsResult:
    T_world_hub: np.ndarray
    T_world_robot_root: np.ndarray
    arm_joint_names: tuple[str, ...]
    arm_joint_values: np.ndarray
    max_position_residual_m: float
    p95_position_residual_m: float
    max_orientation_residual_deg: float
    p95_orientation_residual_deg: float
    max_joint_step_rad: float
    external_ik: Mapping[str, Any]
    ik_constructor_kwargs: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "T_world_hub": self.T_world_hub.tolist(),
            "T_world_robot_root": self.T_world_robot_root.tolist(),
            "arm_joint_names": list(self.arm_joint_names),
            "max_position_residual_m": self.max_position_residual_m,
            "p95_position_residual_m": self.p95_position_residual_m,
            "max_orientation_residual_deg": self.max_orientation_residual_deg,
            "p95_orientation_residual_deg": self.p95_orientation_residual_deg,
            "max_joint_step_rad": self.max_joint_step_rad,
            "external_ik": dict(self.external_ik),
            "ik_constructor_kwargs": dict(self.ik_constructor_kwargs),
            "root_placement": "explicit_shared_T_world_hub",
        }


def build_world_tcp_targets(
    inputs: ParallelJawInputs,
    bundle: RobotBundle,
) -> Mapping[str, np.ndarray]:
    """Apply only the bundle's declared semantic-target-to-TCP rotation."""

    result: dict[str, np.ndarray] = {}
    for side, semantic in (
        ("left", inputs.left_world_semantic),
        ("right", inputs.right_world_semantic),
    ):
        poses = semantic.copy()
        poses[:, :3, :3] = (
            semantic[:, :3, :3]
            @ bundle.semantic_target_to_tcp_rotation[side][None, :, :]
        )
        result[side] = poses
    return result


def build_root_frame_ik_targets(
    inputs: ParallelJawInputs,
    bundle: RobotBundle,
    *,
    T_world_hub: np.ndarray,
) -> tuple[
    np.ndarray, Mapping[str, np.ndarray], list[dict[str, tuple[np.ndarray, np.ndarray]]]
]:
    T_world_robot_root = bundle.world_robot_root(T_world_hub)
    T_robot_root_world = invert_transform(T_world_robot_root)
    world_targets = build_world_tcp_targets(inputs, bundle)
    root_targets = {
        side: T_robot_root_world[None, :, :] @ world_targets[side]
        for side in ("left", "right")
    }
    targets: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for frame in range(inputs.frame_count):
        frame_target: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in ("left", "right"):
            pose = root_targets[side][frame]
            frame_target[bundle.tcp_frames[side]] = (
                pose[:3, 3].copy(),
                matrix_to_quaternion_wxyz(pose[:3, :3]),
            )
        targets.append(frame_target)
    return T_world_robot_root, root_targets, targets


def validate_kinematics_quality(
    *,
    position_residuals_m: np.ndarray,
    orientation_residuals_deg: np.ndarray | None = None,
    joint_values: np.ndarray,
    max_position_residual_m: float,
    max_orientation_residual_deg: float = 20.0,
    max_joint_step_rad: float,
) -> tuple[float, float, float, float, float]:
    residuals = np.asarray(position_residuals_m, dtype=np.float64)
    values = np.asarray(joint_values, dtype=np.float64)
    if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all():
        raise KinematicsError("IK position residuals must be a non-empty finite vector")
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise KinematicsError("IK joint trajectory must be a non-empty finite matrix")
    if (
        not np.isfinite(max_position_residual_m)
        or max_position_residual_m <= 0.0
        or not np.isfinite(max_orientation_residual_deg)
        or max_orientation_residual_deg <= 0.0
        or not np.isfinite(max_joint_step_rad)
        or max_joint_step_rad <= 0.0
    ):
        raise ValueError(
            "IK position/orientation residual and joint-step thresholds must be positive"
        )
    orientations = (
        np.zeros_like(residuals)
        if orientation_residuals_deg is None
        else np.asarray(orientation_residuals_deg, dtype=np.float64)
    )
    if (
        orientations.shape != residuals.shape
        or not np.isfinite(orientations).all()
        or np.any(orientations < 0.0)
    ):
        raise KinematicsError(
            "IK orientation residuals must be a finite nonnegative vector matching "
            "position residuals"
        )
    residual_max = float(residuals.max())
    residual_p95 = float(np.percentile(residuals, 95.0))
    orientation_max = float(orientations.max())
    orientation_p95 = float(np.percentile(orientations, 95.0))
    joint_step = (
        float(np.max(np.abs(np.diff(values, axis=0)))) if values.shape[0] > 1 else 0.0
    )
    if residual_max > max_position_residual_m:
        raise KinematicsError(
            f"dual-arm IK max TCP position residual {residual_max:.6f} m exceeds "
            f"threshold {max_position_residual_m:.6f} m"
        )
    if orientation_max > max_orientation_residual_deg:
        raise KinematicsError(
            f"dual-arm IK max TCP orientation residual {orientation_max:.6f} deg "
            f"exceeds threshold {max_orientation_residual_deg:.6f} deg"
        )
    if joint_step > max_joint_step_rad:
        raise KinematicsError(
            f"dual-arm IK max frame-to-frame arm joint step {joint_step:.6f} rad "
            f"exceeds threshold {max_joint_step_rad:.6f} rad"
        )
    return (
        residual_max,
        residual_p95,
        orientation_max,
        orientation_p95,
        joint_step,
    )


def _validate_arm_joint_limits(
    values: np.ndarray,
    names: tuple[str, ...],
    bundle: RobotBundle,
) -> None:
    tolerance = 1e-5
    for column, joint in enumerate(names):
        for label, inspection in (
            ("ik_urdf", bundle.ik_inspection),
            ("render_urdf", bundle.render_inspection),
        ):
            if joint not in inspection.joint_limits:
                raise KinematicsError(f"{label} lacks arm joint {joint!r}")
            limit = inspection.joint_limits[joint]
            series = values[:, column]
            if limit.lower is not None and np.any(series < limit.lower - tolerance):
                frame = int(np.argmin(series))
                raise KinematicsError(
                    f"{joint}[{frame}]={series[frame]:.8g} is below {label} lower "
                    f"limit {limit.lower:.8g}"
                )
            if limit.upper is not None and np.any(series > limit.upper + tolerance):
                frame = int(np.argmax(series))
                raise KinematicsError(
                    f"{joint}[{frame}]={series[frame]:.8g} is above {label} upper "
                    f"limit {limit.upper:.8g}"
                )


def solve_kinematics(
    inputs: ParallelJawInputs,
    bundle: RobotBundle,
    *,
    scene_utils_root: str | Path,
    T_world_hub: np.ndarray,
    orientation_cost: float = 0.010,
    max_position_residual_m: float = 0.01,
    max_orientation_residual_deg: float = 20.0,
    max_joint_step_rad: float = 0.4,
) -> KinematicsResult:
    """Solve one temporally coherent dual-arm trajectory at a fixed shared hub."""

    T_world_hub = validate_transform(T_world_hub, name="T_world_hub")
    if not np.isfinite(orientation_cost) or orientation_cost <= 0.0:
        raise ValueError("orientation_cost must be positive and finite")
    T_world_robot_root, root_targets, targets = build_root_frame_ik_targets(
        inputs,
        bundle,
        T_world_hub=T_world_hub,
    )
    external = load_arm_ik(scene_utils_root)
    ik_kwargs = {
        **IK_CONSTRUCTOR_KWARGS,
        "orientation_cost": float(orientation_cost),
    }
    ik = external.module.ArmIK(
        str(bundle.ik_urdf),
        flanges=(
            bundle.tcp_frames["left"],
            bundle.tcp_frames["right"],
        ),
        **ik_kwargs,
    )
    ik_names = tuple(str(name) for name in ik.joint_names)
    if set(ik_names) != set(bundle.arm_joint_names) or len(ik_names) != len(
        bundle.arm_joint_names
    ):
        raise KinematicsError(
            "Pink/Pinocchio joints do not exactly cover bundle arm_joint_names; "
            f"pink={ik_names}, bundle={bundle.arm_joint_names}"
        )
    raw_values = np.asarray(ik.solve_trajectory(targets), dtype=np.float64)
    if raw_values.shape != (inputs.frame_count, len(ik_names)):
        raise KinematicsError(
            f"ArmIK returned shape {raw_values.shape}, expected "
            f"{(inputs.frame_count, len(ik_names))}"
        )
    if not np.isfinite(raw_values).all():
        raise KinematicsError("ArmIK returned non-finite joint values")

    reorder = [ik_names.index(name) for name in bundle.arm_joint_names]
    values = raw_values[:, reorder]
    _validate_arm_joint_limits(values, bundle.arm_joint_names, bundle)

    position_residuals: list[float] = []
    orientation_residuals: list[float] = []
    for frame in range(inputs.frame_count):
        ik.configuration.q = raw_values[frame].copy()
        for side in ("left", "right"):
            tcp = bundle.tcp_frames[side]
            actual_position, actual_rotation = ik.flange_pose(tcp)
            target_pose = root_targets[side][frame]
            position_residuals.append(
                float(np.linalg.norm(np.asarray(actual_position) - target_pose[:3, 3]))
            )
            orientation_residuals.append(
                orientation_error_degrees(
                    np.asarray(actual_rotation),
                    target_pose[:3, :3],
                )
            )
    orientation_array = np.asarray(orientation_residuals)
    (
        residual_max,
        residual_p95,
        orientation_max,
        orientation_p95,
        joint_step,
    ) = validate_kinematics_quality(
        position_residuals_m=np.asarray(position_residuals),
        orientation_residuals_deg=orientation_array,
        joint_values=values,
        max_position_residual_m=max_position_residual_m,
        max_orientation_residual_deg=max_orientation_residual_deg,
        max_joint_step_rad=max_joint_step_rad,
    )
    return KinematicsResult(
        T_world_hub=T_world_hub.copy(),
        T_world_robot_root=T_world_robot_root,
        arm_joint_names=bundle.arm_joint_names,
        arm_joint_values=values.astype(np.float32),
        max_position_residual_m=residual_max,
        p95_position_residual_m=residual_p95,
        max_orientation_residual_deg=orientation_max,
        p95_orientation_residual_deg=orientation_p95,
        max_joint_step_rad=joint_step,
        external_ik=external.as_dict(),
        ik_constructor_kwargs=ik_kwargs,
    )
