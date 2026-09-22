# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow capabilities exposed by loaded task references.

A loaded reference exposes common task/object data plus the robot-motion capability
present in its source schema. Arrays are host (numpy) full trajectories; the env
binds and uploads the required data at setup.

Conventions: positions in metres, quaternions ``wxyz``, time index first (``T``),
``B`` object bodies, ``K`` contact points per hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ReferenceMetadata:
    """Source identity kept separate from the resampled playback rate and arrays."""

    source_path: str
    source_fps: float
    is_resampled: bool = False
    source_frame_playback: bool = False
    schema_version: str | None = None
    motion_kind: str | None = None
    source_dataset: str | None = None
    sequence_id: str | None = None
    robot_name: str | None = None
    raw_motion_file: str | None = None
    coord_frame: str | None = None


@dataclass(frozen=True)
class ObjectBodySpec:
    """Map one imported terminal body name to one named reference body."""

    reference_name: str
    simulation_name: str | None

    def __post_init__(self) -> None:
        if not self.reference_name:
            raise ValueError("object reference body names must be nonempty")
        if self.simulation_name == "":
            raise ValueError("object simulation body names must be nonempty when provided")


@dataclass(frozen=True)
class ObjectJointPhysicsSpec:
    """Passive physical properties for one imported scalar object joint.

    These properties remain active when virtual object control is zero. They are
    deliberately separate from :class:`ObjectJointDriveSpec`, whose reference
    tracking authority is scaled by VOC.
    """

    armature: float
    friction: float

    def __post_init__(self) -> None:
        values = (self.armature, self.friction)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("object articulation physics parameters must be finite")
        if self.armature < 0.0 or self.friction < 0.0:
            raise ValueError("object articulation armature and friction must be nonnegative")


@dataclass(frozen=True)
class ObjectJointDriveSpec:
    """Implicit position-velocity drive parameters for one virtually assisted object articulation."""

    kp: float
    kd: float
    effort_limit: float

    def __post_init__(self) -> None:
        values = (self.kp, self.kd, self.effort_limit)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("object articulation drive parameters must be finite")
        if self.kp < 0.0 or self.kd < 0.0 or self.effort_limit <= 0.0:
            raise ValueError("object articulation drive gains must be nonnegative and effort must be positive")


@dataclass(frozen=True)
class ObjectArticulationSpec:
    """Map one imported terminal joint name to one reference articulation column."""

    simulation_joint_name: str
    reference_index: int
    physics: ObjectJointPhysicsSpec
    drive: ObjectJointDriveSpec

    def __post_init__(self) -> None:
        if not self.simulation_joint_name:
            raise ValueError("object articulation joint names must be nonempty")
        if self.reference_index < 0:
            raise ValueError("object articulation reference indices must be nonnegative")


@dataclass(frozen=True)
class ObjectAssetSpec:
    """Explicit setup-time contract between one URDF component and object reference arrays."""

    name: str
    urdf_path: str
    bodies: tuple[ObjectBodySpec, ...]
    root_reference_name: str
    articulations: tuple[ObjectArticulationSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bodies", tuple(self.bodies))
        object.__setattr__(self, "articulations", tuple(self.articulations))
        if not self.name or not self.urdf_path:
            raise ValueError("object asset names and URDF paths must be nonempty")
        if not self.bodies:
            raise ValueError(f"object asset {self.name!r} must bind at least one body")
        reference_names = tuple(body.reference_name for body in self.bodies)
        simulation_names = tuple(body.simulation_name for body in self.bodies if body.simulation_name is not None)
        if len(set(reference_names)) != len(reference_names):
            raise ValueError(f"object asset {self.name!r} has duplicate reference body bindings")
        if len(set(simulation_names)) != len(simulation_names):
            raise ValueError(f"object asset {self.name!r} has duplicate simulation body bindings")
        if any(body.simulation_name is None for body in self.bodies) and len(self.bodies) != 1:
            raise ValueError("unnamed simulation-body binding is valid only for a single-body asset")
        if self.root_reference_name not in reference_names:
            raise ValueError(
                f"object asset {self.name!r} root {self.root_reference_name!r} is not a bound reference body"
            )
        articulation_names = tuple(binding.simulation_joint_name for binding in self.articulations)
        articulation_indices = tuple(binding.reference_index for binding in self.articulations)
        if len(set(articulation_names)) != len(articulation_names):
            raise ValueError(f"object asset {self.name!r} has duplicate articulation joint bindings")
        if len(set(articulation_indices)) != len(articulation_indices):
            raise ValueError(f"object asset {self.name!r} has duplicate articulation reference bindings")


@runtime_checkable
class Reference(Protocol):
    """Common task, object, contact, asset, and timing data."""

    @property
    def num_frames(self) -> int: ...

    @property
    def fps(self) -> float: ...

    @property
    def metadata(self) -> ReferenceMetadata: ...

    @property
    def sides(self) -> tuple[str, ...]:
        """Hand sides present, e.g. ``("left", "right")``."""
        ...

    # --- object reference (consumed by VOC + objectives; embodiment-agnostic) ---
    def object_body_pos_w(self) -> np.ndarray:
        """``(T, B, 3)`` per-body object position [m]."""
        ...

    def object_body_quat_w(self) -> np.ndarray:
        """``(T, B, 4)`` per-body object orientation (wxyz)."""
        ...

    def object_articulation(self) -> np.ndarray:
        """``(T, A)`` articulation joint positions [rad or m]; ``A==0`` if rigid."""
        ...

    # --- contact reference (consumed by contact objective; embodiment-agnostic) ---
    def contact_pos_w(self, side: str) -> np.ndarray:
        """``(T, K, 3)`` reference contact positions [m]."""
        ...

    def contact_normal_w(self, side: str) -> np.ndarray:
        """``(T, K, 3)`` reference contact normals (unit)."""
        ...

    def contact_part_ids(self, side: str) -> np.ndarray:
        """``(T, K)`` object body index each contact point belongs to (1-indexed)."""
        ...

    def contact_active(self, side: str) -> np.ndarray:
        """``(T,)`` binary label indicating whether the hand should contact an object."""
        ...

    # --- object assets (consumed by scene construction) ---
    def object_name(self) -> str:
        """Object identifier (e.g. arctic ``box`` / ``mixer``) — resolves an articulated URDF."""
        ...

    def object_body_names(self) -> list[str]:
        """``B`` object body names (rigid: independent bodies; articulated: parts)."""
        ...

    def object_mesh_paths(self) -> list[str]:
        """``B`` object collision/visual mesh paths (as stored; remap at load)."""
        ...

    def object_urdf_paths(self) -> list[str]:
        """Object URDF paths (per rigid body; empty for a mesh-only articulated object)."""
        ...

    def object_mesh_radius(self) -> np.ndarray:
        """``(B,)`` per-body bounding-ball radius — the wrench-support torque scale ``rc``."""
        ...

    def object_assets(self) -> tuple[ObjectAssetSpec, ...]:
        """Explicit URDF/body/root/articulation bindings for scene construction."""
        ...

    def frame_window(self, start_frame: int = 0, end_frame: int = -1) -> Reference:
        """Return source frames ``[start_frame, end_frame)``; a negative end means sequence end."""
        ...


def reference_frame_slice(num_frames: int, start_frame: int = 0, end_frame: int = -1) -> slice:
    """Validate and resolve a half-open reference-frame window.

    This matches robotic_grounding's motion trimming convention: ``start_frame``
    is inclusive, ``end_frame`` is exclusive, and any negative end selects the
    remainder of the sequence.
    """
    end = num_frames if end_frame < 0 else end_frame
    if start_frame < 0 or end > num_frames or start_frame >= end:
        raise ValueError(
            f"invalid reference frame window [{start_frame}, {end}) for {num_frames} frames; "
            "require 0 <= start_frame < end_frame <= num_frames"
        )
    return slice(start_frame, end)


@runtime_checkable
class HandPoseReference(Reference, Protocol):
    """Legacy pair-of-hands motion represented by wrist poses and finger joints."""

    def wrist_pos_w(self, side: str) -> np.ndarray:
        """``(T, 3)`` wrist position [m]."""
        ...

    def wrist_quat_w(self, side: str) -> np.ndarray:
        """``(T, 4)`` wrist orientation (wxyz)."""
        ...

    def finger_joint_pos(self, side: str) -> np.ndarray:
        """``(T, Nf)`` finger joint positions [rad]."""
        ...

    def finger_joint_names(self, side: str) -> list[str]:
        """Names corresponding to the last axis of :meth:`finger_joint_pos`."""
        ...

    def frame_pos_w(self, side: str) -> np.ndarray:
        """``(T, F, 3)`` robot task-frame positions [m]."""
        ...

    def frame_names(self, side: str) -> list[str]:
        """``F`` robot task-frame names (select fingertip frames by hand-body name)."""
        ...


@runtime_checkable
class NamedRobotReference(Reference, Protocol):
    """Whole-robot motion represented by named scalar joint coordinates."""

    def robot_joint_pos(self) -> np.ndarray:
        """``(T, J)`` joint positions in :meth:`robot_joint_names` order."""
        ...

    def robot_joint_names(self) -> list[str]:
        """Unique names corresponding to the last axis of :meth:`robot_joint_pos`."""
        ...

    def robot_root_pos_w(self) -> np.ndarray:
        """``(T, 3)`` robot root position [m]."""
        ...

    def robot_root_quat_w(self) -> np.ndarray:
        """``(T, 4)`` robot root orientation (wxyz)."""
        ...


@runtime_checkable
class NamedFrameReference(Reference, Protocol):
    """Named robot-frame poses exported for validation or task semantics."""

    def robot_frame_names(self) -> list[str]:
        """Unique names corresponding to the second axis of the frame-pose arrays."""
        ...

    def robot_frame_pos_w(self) -> np.ndarray:
        """``(T, F, 3)`` named robot-frame positions [m]."""
        ...

    def robot_frame_quat_w(self) -> np.ndarray:
        """``(T, F, 4)`` named robot-frame orientations (wxyz)."""
        ...
