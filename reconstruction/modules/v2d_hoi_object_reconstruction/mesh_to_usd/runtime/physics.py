# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python helpers used by mesh authoring and the Isaac Sim drop test."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StandingClassification:
    standing: bool
    tilt_degrees: float
    touching_ground: bool


@dataclass(frozen=True)
class DropTestOutcome:
    rigid_body_test_passed: bool
    standing_requirement_passed: bool
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass
class SettleTracker:
    """Track consecutive low-motion frames after the first observed contact."""

    required_frames: int
    linear_speed_limit: float
    angular_speed_limit: float
    contact_seen: bool = False
    consecutive_frames: int = 0

    def __post_init__(self) -> None:
        if self.required_frames <= 0:
            raise ValueError("required settle frames must be positive")
        limits = (self.linear_speed_limit, self.angular_speed_limit)
        if not all(math.isfinite(value) and value >= 0 for value in limits):
            raise ValueError("settle speed limits must be finite and non-negative")
        if self.consecutive_frames < 0:
            raise ValueError("consecutive settle frames cannot be negative")

    @property
    def settled(self) -> bool:
        return self.consecutive_frames >= self.required_frames

    def observe(
        self,
        *,
        contact_active: bool,
        linear_speed: float,
        angular_speed: float,
    ) -> bool:
        """Consume one frame and return whether the body is currently settled."""

        speeds = (float(linear_speed), float(angular_speed))
        if not all(math.isfinite(value) and value >= 0 for value in speeds):
            raise ValueError("observed speeds must be finite and non-negative")
        self.contact_seen = self.contact_seen or bool(contact_active)
        below_limits = (
            speeds[0] < self.linear_speed_limit
            and speeds[1] < self.angular_speed_limit
        )
        self.consecutive_frames = (
            self.consecutive_frames + 1
            if self.contact_seen and below_limits
            else 0
        )
        return self.settled


def quaternion_angular_distance_degrees(
    first_wxyz: Sequence[float],
    second_wxyz: Sequence[float],
) -> float:
    """Return the shortest angular distance between two quaternions."""

    first = np.asarray(first_wxyz, dtype=np.float64)
    second = np.asarray(second_wxyz, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("quaternions must contain 4 values")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("quaternions must be finite")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("quaternions cannot be zero")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


@dataclass
class PoseSettleTracker:
    """Track a bounded full-pose stability window after first ground contact."""

    required_frames: int
    position_tolerance: float
    angular_tolerance_degrees: float
    contact_seen: bool = False
    positions: list[np.ndarray] = field(default_factory=list, repr=False)
    orientations: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.required_frames <= 0:
            raise ValueError("required settle frames must be positive")
        thresholds = (self.position_tolerance, self.angular_tolerance_degrees)
        if not all(math.isfinite(value) and value > 0 for value in thresholds):
            raise ValueError("pose settle thresholds must be positive and finite")

    @property
    def position_span(self) -> float | None:
        if not self.positions:
            return None
        return max(
            (
                float(np.linalg.norm(first - second))
                for index, first in enumerate(self.positions)
                for second in self.positions[index + 1 :]
            ),
            default=0.0,
        )

    @property
    def angular_span_degrees(self) -> float | None:
        if not self.orientations:
            return None
        return max(
            (
                quaternion_angular_distance_degrees(first, second)
                for index, first in enumerate(self.orientations)
                for second in self.orientations[index + 1 :]
            ),
            default=0.0,
        )

    @property
    def settled(self) -> bool:
        if len(self.positions) < self.required_frames:
            return False
        position_span = self.position_span
        angular_span = self.angular_span_degrees
        return bool(
            position_span is not None
            and angular_span is not None
            and position_span <= self.position_tolerance
            and angular_span <= self.angular_tolerance_degrees
        )

    def observe(
        self,
        *,
        contact_active: bool,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
    ) -> bool:
        """Consume one pose sample and return whether its window is stable."""

        position_array = np.asarray(position, dtype=np.float64)
        orientation_array = np.asarray(orientation_wxyz, dtype=np.float64)
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            raise ValueError("position must contain 3 finite values")
        quaternion_angular_distance_degrees(orientation_array, orientation_array)
        self.contact_seen = self.contact_seen or bool(contact_active)
        if not self.contact_seen:
            self.positions.clear()
            self.orientations.clear()
            return False
        self.positions.append(position_array)
        self.orientations.append(orientation_array)
        if len(self.positions) > self.required_frames:
            self.positions.pop(0)
            self.orientations.pop(0)
        return self.settled


def classify_drop_test_outcome(
    *,
    contact_seen: bool,
    settled: bool,
    timed_out: bool,
    standing: bool,
    standing_required: bool,
) -> DropTestOutcome:
    """Classify the core rigid-body test and optional standing requirement."""

    rigid_body_test_passed = bool(contact_seen and settled and not timed_out)
    standing_requirement_passed = bool(not standing_required or standing)
    failure_reasons = []
    if not contact_seen:
        failure_reasons.append("ground_contact_not_detected")
    if not settled:
        failure_reasons.append("object_did_not_settle")
    if timed_out:
        failure_reasons.append("drop_timed_out")
    if standing_required and not standing:
        failure_reasons.append("optional_standing_requirement_not_met")
    return DropTestOutcome(
        rigid_body_test_passed=rigid_body_test_passed,
        standing_requirement_passed=standing_requirement_passed,
        passed=rigid_body_test_passed and standing_requirement_passed,
        failure_reasons=tuple(failure_reasons),
    )


def aggregate_pose_results(
    pose_results: Sequence[dict],
    *,
    standing_required: bool,
) -> dict:
    """Summarize a pose sweep, passing when any candidate satisfies the checks."""

    if not pose_results:
        raise ValueError("pose_results cannot be empty")

    rigid_body_pass_count = sum(
        bool(result.get("rigid_body_test_passed")) for result in pose_results
    )
    standing_pass_count = sum(bool(result.get("standing")) for result in pose_results)
    passed_count = sum(
        bool(result.get("rigid_body_test_passed"))
        and (not standing_required or bool(result.get("standing")))
        for result in pose_results
    )

    def rank(index: int) -> tuple[bool, bool, bool, float, int]:
        result = pose_results[index]
        tilt = float(result.get("tilt_degrees", math.inf))
        if not math.isfinite(tilt):
            tilt = math.inf
        return (
            bool(result.get("rigid_body_test_passed"))
            and (not standing_required or bool(result.get("standing"))),
            bool(result.get("standing")),
            bool(result.get("rigid_body_test_passed")),
            -tilt,
            -index,
        )

    representative_index = max(range(len(pose_results)), key=rank)
    representative = pose_results[representative_index]
    failure_reasons = []
    if rigid_body_pass_count == 0:
        failure_reasons.append("no_pose_candidate_passed_rigid_body_test")
    if standing_required and standing_pass_count == 0:
        failure_reasons.append("no_pose_candidate_met_standing_requirement")
    if (
        standing_required
        and rigid_body_pass_count > 0
        and standing_pass_count > 0
        and passed_count == 0
    ):
        failure_reasons.append("no_pose_candidate_met_all_required_checks")

    passed = passed_count > 0
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "rigid_body_test_passed": rigid_body_pass_count > 0,
        "standing": standing_pass_count > 0,
        "standing_required": bool(standing_required),
        "standing_requirement_passed": bool(
            not standing_required or passed_count > 0
        ),
        "failure_reasons": failure_reasons,
        "candidate_count": len(pose_results),
        "rigid_body_pass_count": rigid_body_pass_count,
        "standing_pass_count": standing_pass_count,
        "passed_candidate_count": passed_count,
        "representative_candidate_index": representative_index,
        "representative_pose_id": representative.get("pose_id"),
        "tilt_degrees": representative.get("tilt_degrees"),
    }


def physics_contact_config(characteristic_extent_m: float) -> dict[str, float | int | bool]:
    """Return the pinned small-object contact and rigid-solver configuration."""

    extent = float(characteristic_extent_m)
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("characteristic_extent_m must be positive and finite")
    return {
        "rest_offset_m": 0.0,
        "contact_offset_m": max(0.001, min(0.005, 0.02 * extent)),
        "solver_position_iterations": 32,
        "solver_velocity_iterations": 4,
        "ccd_enabled": True,
    }


def ground_metrics(
    minimum: Sequence[float],
    maximum: Sequence[float],
    ground_z: float,
) -> dict[str, float]:
    """Describe signed visual grounding using exact world-space mesh bounds."""

    if len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("minimum and maximum must be 3-vectors")
    values = tuple(float(value) for value in (*minimum, *maximum, ground_z))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ground metric inputs must be finite")
    extents = tuple(float(maximum[i]) - float(minimum[i]) for i in range(3))
    characteristic_extent = max(extents)
    if characteristic_extent <= 0:
        raise ValueError("mesh bounds must have a positive extent")
    clearance = float(minimum[2]) - float(ground_z)
    gap = max(clearance, 0.0)
    penetration = max(-clearance, 0.0)
    return {
        "minimum_z": float(minimum[2]),
        "clearance": clearance,
        "distance": abs(clearance),
        "gap": gap,
        "penetration": penetration,
        "gap_ratio": gap / characteristic_extent,
        "penetration_ratio": penetration / characteristic_extent,
        "characteristic_extent": characteristic_extent,
    }


def is_within_ground_tolerance(clearance: float, tolerance: float) -> bool:
    """Return whether signed ground clearance is within an allowed distance."""

    clearance = float(clearance)
    tolerance = float(tolerance)
    if not math.isfinite(clearance) or not math.isfinite(tolerance):
        raise ValueError("ground clearance and tolerance must be finite")
    if tolerance < 0:
        raise ValueError("ground tolerance cannot be negative")
    return abs(clearance) <= tolerance


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def quaternion_rotate(
    vector: Sequence[float], quaternion_wxyz: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a 3-vector by a scalar-first quaternion."""
    if len(vector) != 3 or len(quaternion_wxyz) != 4:
        raise ValueError("Expected a 3-vector and a wxyz quaternion")
    w = quaternion_wxyz[0]
    xyz = quaternion_wxyz[1:]
    t = tuple(2.0 * value for value in _cross(xyz, vector))
    xyz_cross_t = _cross(xyz, t)
    return tuple(vector[i] + w * t[i] + xyz_cross_t[i] for i in range(3))


def tilt_from_vertical_degrees(
    quaternion_wxyz: Sequence[float],
    local_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    norm = math.sqrt(sum(value * value for value in local_up))
    if norm == 0:
        raise ValueError("local_up cannot be the zero vector")
    normalized_up = tuple(value / norm for value in local_up)
    world_up = quaternion_rotate(normalized_up, quaternion_wxyz)
    world_norm = math.sqrt(sum(value * value for value in world_up))
    cosine = max(-1.0, min(1.0, world_up[2] / world_norm))
    return math.degrees(math.acos(cosine))


def classify_standing(
    *,
    quaternion_wxyz: Sequence[float],
    settled: bool,
    contact_seen: bool,
    touching_ground: bool,
    max_tilt_degrees: float,
    local_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> StandingClassification:
    tilt = tilt_from_vertical_degrees(quaternion_wxyz, local_up)
    return StandingClassification(
        standing=bool(
            settled
            and contact_seen
            and touching_ground
            and tilt <= max_tilt_degrees
        ),
        tilt_degrees=tilt,
        touching_ground=touching_ground,
    )
