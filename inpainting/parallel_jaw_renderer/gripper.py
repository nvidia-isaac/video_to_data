"""Map physical inner-jaw aperture to embodiment-specific URDF joints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .bundle import GripperMappingSpec, UrdfInspection


class GripperMappingError(ValueError):
    """Raised when a physical aperture cannot be mapped unambiguously."""


@dataclass(frozen=True)
class GripperTrajectory:
    names: tuple[str, ...]
    values: np.ndarray
    report: Mapping[str, Any]


def _finite_number(params: Mapping[str, Any], name: str) -> float:
    value = params.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GripperMappingError(f"gripper_mapping.params.{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise GripperMappingError(f"gripper_mapping.params.{name} must be finite")
    return result


def _side_vector(
    value: object,
    *,
    side: str,
    size: int,
    name: str,
) -> np.ndarray:
    if isinstance(value, dict):
        if set(value) != {"left", "right"}:
            raise GripperMappingError(
                f"gripper_mapping.params.{name} side map must contain left and right"
            )
        value = value[side]
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        array = np.full(size, float(array), dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise GripperMappingError(
            f"gripper_mapping.params.{name} for {side} must be scalar or shape ({size},)"
        )
    return array


def _validate_joint_values(
    values: np.ndarray,
    names: tuple[str, ...],
    inspection: UrdfInspection,
) -> None:
    tolerance = 1e-6
    for column, joint in enumerate(names):
        if joint not in inspection.joint_limits:
            raise GripperMappingError(f"render URDF lacks gripper joint {joint!r}")
        limit = inspection.joint_limits[joint]
        series = values[:, column]
        if limit.lower is not None and np.any(series < limit.lower - tolerance):
            frame = int(np.argmin(series))
            raise GripperMappingError(
                f"mapped {joint}[{frame}]={series[frame]:.8g} is below "
                f"URDF lower limit {limit.lower:.8g}"
            )
        if limit.upper is not None and np.any(series > limit.upper + tolerance):
            frame = int(np.argmax(series))
            raise GripperMappingError(
                f"mapped {joint}[{frame}]={series[frame]:.8g} is above "
                f"URDF upper limit {limit.upper:.8g}"
            )


def _galbot_four_bar(
    aperture_m: np.ndarray,
    *,
    side: str,
    spec: GripperMappingSpec,
) -> GripperTrajectory:
    names = spec.joint_names[side]
    if len(names) != 1:
        raise GripperMappingError(
            f"galbot_four_bar requires one driven joint for {side}; mimic followers "
            "must be left to yourdfpy"
        )
    params = spec.params
    half_gap = _finite_number(params, "inner_pivot_half_gap_m")
    pad_inset = _finite_number(params, "pad_inset_m")
    link_length = _finite_number(params, "finger_link_length_m")
    knuckle = _finite_number(params, "knuckle_angle_rad")
    lower = _finite_number(params, "joint_lower_rad")
    upper = _finite_number(params, "joint_upper_rad")
    if link_length <= 0.0 or lower >= upper:
        raise GripperMappingError(
            "galbot_four_bar needs positive finger_link_length_m and "
            "joint_lower_rad < joint_upper_rad"
        )

    # Source linkage:
    # opening = 2 * (half_gap - pad_inset + L*sin(knuckle-q)).
    endpoint_openings = 2.0 * (
        half_gap
        - pad_inset
        + link_length * np.sin(knuckle - np.asarray((lower, upper)))
    )
    aperture_min = max(0.0, float(endpoint_openings.min()))
    aperture_max = float(endpoint_openings.max())
    clipped = np.clip(aperture_m, aperture_min, aperture_max)
    sine = (0.5 * clipped - (half_gap - pad_inset)) / link_length
    joint = knuckle - np.arcsin(np.clip(sine, -1.0, 1.0))
    joint = np.clip(joint, lower, upper)
    values = joint[:, None]
    report = {
        "kind": spec.kind,
        "side": side,
        "formula": (
            "q=knuckle_angle-arcsin((aperture/2-"
            "(inner_pivot_half_gap-pad_inset))/finger_link_length)"
        ),
        "input_aperture_m_range": [
            float(aperture_m.min()),
            float(aperture_m.max()),
        ],
        "supported_aperture_m_range": [aperture_min, aperture_max],
        "clipped_frame_count": int(np.count_nonzero(clipped != aperture_m)),
        "joint_range": [float(joint.min()), float(joint.max())],
        "mimic_policy": "command_parent_only_yourdfpy_expands_urdf_mimics",
    }
    return GripperTrajectory(names=names, values=values, report=report)


def _mirrored_prismatic(
    aperture_m: np.ndarray,
    *,
    side: str,
    spec: GripperMappingSpec,
) -> GripperTrajectory:
    names = spec.joint_names[side]
    if len(names) != 2:
        raise GripperMappingError(
            f"mirrored_prismatic requires two finger joints for {side}"
        )
    params = spec.params
    closed_aperture = _finite_number(params, "closed_aperture_m")
    open_aperture = _finite_number(params, "open_aperture_m")
    if closed_aperture < 0.0 or open_aperture <= closed_aperture:
        raise GripperMappingError(
            "mirrored_prismatic needs 0 <= closed_aperture_m < open_aperture_m"
        )
    closed_joint = _side_vector(
        params.get("closed_joint_position_m"),
        side=side,
        size=len(names),
        name="closed_joint_position_m",
    )
    open_joint = _side_vector(
        params.get("open_joint_position_m"),
        side=side,
        size=len(names),
        name="open_joint_position_m",
    )
    clipped = np.clip(aperture_m, closed_aperture, open_aperture)
    fraction = (clipped - closed_aperture) / (open_aperture - closed_aperture)
    values = (
        closed_joint[None, :] + fraction[:, None] * (open_joint - closed_joint)[None, :]
    )
    report = {
        "kind": spec.kind,
        "side": side,
        "formula": "linear aperture interpolation to two URDF prismatic joints",
        "input_aperture_m_range": [
            float(aperture_m.min()),
            float(aperture_m.max()),
        ],
        "supported_aperture_m_range": [closed_aperture, open_aperture],
        "clipped_frame_count": int(np.count_nonzero(clipped != aperture_m)),
        "joint_min": values.min(axis=0).tolist(),
        "joint_max": values.max(axis=0).tolist(),
        "mirror_policy": "URDF joint axes encode opposing finger directions",
    }
    return GripperTrajectory(names=names, values=values, report=report)


def map_aperture_trajectory(
    aperture_m: np.ndarray,
    *,
    side: str,
    spec: GripperMappingSpec,
    render_inspection: UrdfInspection,
) -> GripperTrajectory:
    aperture = np.asarray(aperture_m, dtype=np.float64)
    if aperture.ndim != 1 or aperture.size == 0:
        raise GripperMappingError("aperture trajectory must be a non-empty vector")
    if not np.isfinite(aperture).all() or np.any(aperture < 0.0):
        raise GripperMappingError("aperture trajectory must be finite and nonnegative")
    if side not in ("left", "right"):
        raise ValueError(f"unsupported side {side!r}")
    if spec.kind == "galbot_four_bar":
        result = _galbot_four_bar(aperture, side=side, spec=spec)
    elif spec.kind == "mirrored_prismatic":
        result = _mirrored_prismatic(aperture, side=side, spec=spec)
    else:
        raise GripperMappingError(f"unsupported mapping kind {spec.kind!r}")
    _validate_joint_values(result.values, result.names, render_inspection)
    return result
