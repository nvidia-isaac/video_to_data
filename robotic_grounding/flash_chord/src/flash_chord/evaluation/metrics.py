# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpretable CPU success metrics for object-tracking rollouts.

Poses are ``[position, quaternion]`` rows of seven values with **xyzw** quaternion layout;
metres and radians unless a name says otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

ADD_VERTEX_COUNT = 500
ADD_VERTEX_SEED = 0
ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
SPIDER_POSITION_THRESHOLD_M = 0.1
SPIDER_ORIENTATION_THRESHOLD_RAD = 0.5
MANIPTRANS_POSITION_THRESHOLD_M = 0.03
MANIPTRANS_ORIENTATION_THRESHOLD_DEG = 30.0
MPPE_KEYPOINT_RADIUS_M = 0.05

TRACKING_ERROR_NAMES = (
    "wrist_position_m",
    "wrist_orientation_rad",
    "object_position_m",
    "object_orientation_rad",
)
TRACKING_CAUSE_NAMES = ("wrist", "object")
_CAUSE_TERMS = ((0, 1), (2, 3))

_POSE_WIDTH = 7


@dataclass(frozen=True, slots=True)
class TrackingThresholds:
    """Tracking-failure limits in the order of :data:`TRACKING_ERROR_NAMES`; ``None`` disables a term."""

    wrist_position_m: float | None = None
    wrist_orientation_rad: float | None = None
    object_position_m: float | None = None
    object_orientation_rad: float | None = None

    def as_tuple(self) -> tuple[float | None, ...]:
        return (
            self.wrist_position_m,
            self.wrist_orientation_rad,
            self.object_position_m,
            self.object_orientation_rad,
        )

    def gated_causes(self) -> tuple[str, ...]:
        """Cause names holding at least one active limit."""
        limits = self.as_tuple()
        return tuple(
            name
            for cause, name in enumerate(TRACKING_CAUSE_NAMES)
            if any(limits[term] is not None for term in _CAUSE_TERMS[cause])
        )


class CompletionLike(Protocol):
    """Per-world episode outcome consumed by the success metrics."""

    completion_step: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    reference_progress: np.ndarray


@dataclass(frozen=True, slots=True)
class ObjectTrackingMetrics:
    """Cohort scores and the error statistics each threshold was applied to.

    Object errors are the cohort mean of each world's frame-averaged error over object bodies;
    their standard deviations are across worlds, while the ADD deviation pools worlds and frames.
    """

    chord_sr: float
    gated_causes: tuple[str, ...]
    termination_cause_fraction: Mapping[str, float]
    object_position_error_m: float
    object_position_error_std_m: float
    object_orientation_error_rad: float
    object_orientation_error_std_rad: float
    object_orientation_error_deg: float
    object_orientation_error_std_deg: float
    object_position_error_per_body_m: tuple[float, ...]
    object_orientation_error_per_body_rad: tuple[float, ...]
    add_auc: float
    add_auc_per_body: tuple[float, ...]
    mean_add_m: float
    mppe_cm: float
    add_std_m: float
    add_std_per_body_m: tuple[float, ...]
    spider_sr_uncentered: float
    spider_position_error_centered_m: float
    maniptrans_sr: float
    position_mean_centered: bool
    world_count: int
    step_count: int
    object_body_names: tuple[str, ...]


def _pose_array(values: np.ndarray, name: str, rank: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != rank or result.shape[-1] != _POSE_WIDTH:
        raise ValueError(f"{name} must have shape [..., {_POSE_WIDTH}] with rank {rank}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _aligned_poses(achieved: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate an ``(T, W, B, 7)`` cohort against its shared ``(T, B, 7)`` reference."""
    achieved_poses = _pose_array(achieved, "achieved_pose_w", 4)
    reference_poses = _pose_array(reference, "reference_pose_w", 3)
    steps, _, bodies, _ = achieved_poses.shape
    if reference_poses.shape != (steps, bodies, _POSE_WIDTH):
        raise ValueError(
            f"reference_pose_w must have shape {(steps, bodies, _POSE_WIDTH)}, got {reference_poses.shape}"
        )
    return achieved_poses, reference_poses[:, None]


def _body_object_ids(values: np.ndarray, bodies: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.shape[0] != bodies or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"body_object_ids must be {bodies} integers, got {result.dtype}/{result.shape}")
    return result


def quaternion_geodesic_angle(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Shortest-arc angle in ``[0, pi]`` radians between broadcastable xyzw quaternion arrays."""
    # sin(theta/2) form: 2*arccos(dot) loses half its digits near identity.
    dot = np.clip(np.abs(np.sum(a_xyzw * b_xyzw, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def _rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrices ``[..., 3, 3]`` for an xyzw quaternion array."""
    x, y, z, w = (quat_xyzw[..., index] for index in range(4))
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*quat_xyzw.shape[:-1], 3, 3)


def _tracking_error(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 3 or result.shape[2] != len(TRACKING_ERROR_NAMES):
        raise ValueError(
            f"tracking_error must have shape [step, world, {len(TRACKING_ERROR_NAMES)}], got {result.shape}"
        )
    return result


def tracking_cause_masks(tracking_error: np.ndarray, thresholds: TrackingThresholds) -> np.ndarray:
    """``(W, 2)`` masks for whether each named cause was ever crossed.

    ``tracking_error`` is ``(T, W, 4)`` ordered by :data:`TRACKING_ERROR_NAMES`, already reduced
    over hands and objects.
    """
    errors = _tracking_error(tracking_error)
    limits = thresholds.as_tuple()
    causes = np.zeros((errors.shape[1], len(TRACKING_CAUSE_NAMES)), dtype=np.bool_)
    for cause, terms in enumerate(_CAUSE_TERMS):
        for term in terms:
            limit = limits[term]
            if limit is not None:
                causes[:, cause] |= np.any(errors[:, :, term] > limit, axis=0)
    return causes


def chord_success(
    tracking_error: np.ndarray,
    thresholds: TrackingThresholds,
    completion: CompletionLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-world success mask and ``(W, 2)`` cause masks for reaching the reference end uncrossed.

    A self-terminated rollout is rejected: its post-termination errors describe a world held at
    its failure frame rather than the sequence.
    """
    causes = tracking_cause_masks(tracking_error, thresholds)
    worlds = causes.shape[0]
    terminated = np.asarray(completion.terminated, dtype=np.bool_)
    truncated = np.asarray(completion.truncated, dtype=np.bool_)
    progress = np.asarray(completion.reference_progress, dtype=np.float64)
    for name, values in (
        ("terminated", terminated),
        ("truncated", truncated),
        ("reference_progress", progress),
    ):
        if values.shape != (worlds,):
            raise ValueError(f"completion.{name} must have shape {(worlds,)}, got {values.shape}")
    if terminated.any():
        raise ValueError(
            "evaluation rollout must disable the tracking-failure termination terms; "
            f"{int(terminated.sum())} of {worlds} worlds terminated"
        )
    reached_end = truncated & np.isclose(progress, 1.0, rtol=0.0, atol=1.0e-6)
    return reached_end & ~causes.any(axis=-1), causes


def object_add(
    achieved_pose_w: np.ndarray,
    reference_pose_w: np.ndarray,
    object_vertices_o: np.ndarray,
    *,
    world_chunk: int = 32,
) -> np.ndarray:
    """Average 3D distance ``(T, W, B)`` in metres over each body's object-frame vertices.

    ``world_chunk`` bounds the transient ``(T, chunk, V, 3)`` buffer.
    """
    if world_chunk <= 0:
        raise ValueError(f"world_chunk must be positive, got {world_chunk}")
    achieved, reference = _aligned_poses(achieved_pose_w, reference_pose_w)
    steps, worlds, bodies, _ = achieved.shape
    vertices = np.asarray(object_vertices_o, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[0] != bodies or vertices.shape[2] != 3 or vertices.shape[1] == 0:
        raise ValueError(f"object_vertices_o must have shape [{bodies}, vertices, 3], got {vertices.shape}")

    achieved_rotation = _rotation_matrix(achieved[..., 3:])
    reference_rotation = _rotation_matrix(reference[..., 3:])
    delta_rotation = achieved_rotation - reference_rotation
    delta_position = achieved[..., :3] - reference[..., :3]

    add = np.empty((steps, worlds, bodies), dtype=np.float64)
    for body in range(bodies):
        body_vertices = vertices[body]
        for start in range(0, worlds, world_chunk):
            stop = min(start + world_chunk, worlds)
            offsets = np.einsum(
                "twij,vj->twvi",
                delta_rotation[:, start:stop, body],
                body_vertices,
            )
            offsets += delta_position[:, start:stop, body, None, :]
            add[:, start:stop, body] = np.linalg.norm(offsets, axis=-1).mean(axis=-1)
    return add


def object_mppe_cm(achieved_pose_w: np.ndarray, reference_pose_w: np.ndarray, *, world_chunk: int = 32) -> float:
    """Mean per-frame pose error in cm using six body-local keypoints at +/-5 cm on X/Y/Z.

    Average Euclidean keypoint distances within each body, take the maximum across bodies
    per frame and world, then average over all frames and worlds without position centering.
    """
    achieved, _ = _aligned_poses(achieved_pose_w, reference_pose_w)
    steps, worlds, bodies, _ = achieved.shape
    if not steps or not worlds or not bodies:
        raise ValueError("MPPE requires at least one frame, finite world, and object body")
    keypoints = MPPE_KEYPOINT_RADIUS_M * np.concatenate((np.eye(3), -np.eye(3)))
    per_body = object_add(
        achieved,
        reference_pose_w,
        np.broadcast_to(keypoints, (bodies, 6, 3)),
        world_chunk=world_chunk,
    )
    return float(100.0 * per_body.max(axis=-1).mean())


def add_auc(add: np.ndarray) -> tuple[float, tuple[float, ...]]:
    """Mean and per-body area under the ADD accuracy curve, pooled over worlds and frames.

    Accuracy is the fraction of ``(frame, world)`` samples below each threshold; the curve is
    integrated against thresholds rescaled to ``[0, 1]``, so the result is itself in ``[0, 1]``.
    """
    values = np.asarray(add, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"add must have shape [step, world, body], got {values.shape}")
    axis = np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size)
    per_body = tuple(
        float(np.trapezoid(np.array([np.mean(values[..., body] < t) for t in ADD_THRESHOLDS_M]), x=axis))
        for body in range(values.shape[2])
    )
    return float(np.mean(per_body)), per_body


def _per_world_pose_error(achieved: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame-averaged ``(W, B)`` position error in metres and orientation error in radians."""
    position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1).mean(axis=0)
    orientation = quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:]).mean(axis=0)
    return position, orientation


def spider_success(
    achieved_pose_w: np.ndarray,
    reference_pose_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-world success mask, position error in metres, and orientation error in radians.

    Position error is raw; the originating implementation's time-mean subtraction is omitted.
    """
    achieved, reference = _aligned_poses(achieved_pose_w, reference_pose_w)
    position, orientation = _per_world_pose_error(achieved, reference)
    position_error = position.mean(axis=-1)
    orientation_error = orientation.mean(axis=-1)
    success = (position_error <= SPIDER_POSITION_THRESHOLD_M) & (orientation_error <= SPIDER_ORIENTATION_THRESHOLD_RAD)
    return success, position_error, orientation_error


def spider_centered_position_error(achieved_pose_w: np.ndarray, reference_pose_w: np.ndarray) -> np.ndarray:
    """Per-world position error in metres with each trajectory's time mean removed first."""
    achieved, reference = _aligned_poses(achieved_pose_w, reference_pose_w)
    error = achieved[..., :3] - reference[..., :3]
    centered = error - error.mean(axis=0, keepdims=True)
    return np.linalg.norm(centered, axis=-1).mean(axis=0).mean(axis=-1)


def maniptrans_success(
    achieved_pose_w: np.ndarray,
    reference_pose_w: np.ndarray,
    body_object_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-world success mask, position error in metres, and orientation error in degrees.

    Bodies are averaged within each object, then every object must clear both thresholds.
    """
    achieved, reference = _aligned_poses(achieved_pose_w, reference_pose_w)
    object_ids = _body_object_ids(body_object_ids, achieved.shape[2])
    position, orientation = _per_world_pose_error(achieved, reference)
    orientation_deg = np.degrees(orientation)

    success = np.ones(achieved.shape[1], dtype=np.bool_)
    for object_id in np.unique(object_ids):
        members = object_ids == object_id
        success &= (position[:, members].mean(axis=-1) < MANIPTRANS_POSITION_THRESHOLD_M) & (
            orientation_deg[:, members].mean(axis=-1) < MANIPTRANS_ORIENTATION_THRESHOLD_DEG
        )
    return success, position.mean(axis=-1), orientation_deg.mean(axis=-1)


def compute_object_tracking_metrics(
    achieved_pose_w: np.ndarray,
    reference_pose_w: np.ndarray,
    object_vertices_o: np.ndarray,
    body_object_ids: np.ndarray,
    tracking_error: np.ndarray,
    thresholds: TrackingThresholds,
    completion: CompletionLike,
    object_body_names: Sequence[str],
    *,
    world_chunk: int = 32,
) -> ObjectTrackingMetrics:
    """Score one cohort of full-sequence attempts against all four success measures."""
    achieved, _ = _aligned_poses(achieved_pose_w, reference_pose_w)
    steps, worlds, bodies, _ = achieved.shape
    names = tuple(object_body_names)
    if len(names) != bodies:
        raise ValueError(f"object_body_names must name {bodies} bodies, got {len(names)}")
    errors = _tracking_error(tracking_error)
    if errors.shape[:2] != (steps, worlds):
        raise ValueError(f"tracking_error must cover {(steps, worlds)} step-world pairs, got {errors.shape[:2]}")

    success, causes = chord_success(tracking_error, thresholds, completion)
    gated = thresholds.gated_causes()
    add = object_add(achieved_pose_w, reference_pose_w, object_vertices_o, world_chunk=world_chunk)
    auc, auc_per_body = add_auc(add)
    per_body_position, per_body_orientation = _per_world_pose_error(*_aligned_poses(achieved_pose_w, reference_pose_w))
    spider_mask, spider_position, spider_orientation = spider_success(achieved_pose_w, reference_pose_w)
    maniptrans_mask, _, _ = maniptrans_success(achieved_pose_w, reference_pose_w, body_object_ids)
    return ObjectTrackingMetrics(
        chord_sr=float(np.mean(success)),
        gated_causes=gated,
        termination_cause_fraction={
            name: float(np.mean(causes[:, TRACKING_CAUSE_NAMES.index(name)])) for name in gated
        },
        object_position_error_m=float(np.mean(spider_position)),
        object_position_error_std_m=float(np.std(spider_position)),
        object_orientation_error_rad=float(np.mean(spider_orientation)),
        object_orientation_error_std_rad=float(np.std(spider_orientation)),
        object_orientation_error_deg=float(np.degrees(np.mean(spider_orientation))),
        object_orientation_error_std_deg=float(np.degrees(np.std(spider_orientation))),
        object_position_error_per_body_m=tuple(float(value) for value in per_body_position.mean(axis=0)),
        object_orientation_error_per_body_rad=tuple(float(value) for value in per_body_orientation.mean(axis=0)),
        add_auc=auc,
        add_auc_per_body=auc_per_body,
        mean_add_m=float(np.mean(add)),
        mppe_cm=object_mppe_cm(achieved_pose_w, reference_pose_w, world_chunk=world_chunk),
        add_std_m=float(np.std(add)),
        add_std_per_body_m=tuple(float(np.std(add[..., body])) for body in range(add.shape[2])),
        spider_sr_uncentered=float(np.mean(spider_mask)),
        spider_position_error_centered_m=float(
            np.mean(spider_centered_position_error(achieved_pose_w, reference_pose_w))
        ),
        maniptrans_sr=float(np.mean(maniptrans_mask)),
        position_mean_centered=False,
        world_count=worlds,
        step_count=steps,
        object_body_names=names,
    )
