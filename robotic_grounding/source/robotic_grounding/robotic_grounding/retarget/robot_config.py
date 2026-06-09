# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Per-robot retarget config loader.

Reads a two-file bundle per robot:

* ``configs/<robot>/frame_alignment.json`` -> world axis swap + per-bone
  rotation/translation corrections that map the source skeleton frame
  into the robot frame.
* ``configs/<robot>/retargeter.json`` -> URDF, IK map, foot framing,
  optional posture-task weights.

Files of either shape can be supplied to :func:`load_robot_config`, which
returns a :class:`RobotRetargetConfig` whose ``r_per_bone`` and
``r_per_link`` dicts already hold the final composed numpy 3x3 matrices the
IK runtime expects. Runtime code does not need to know whether a value was
supplied as ``q_offset_xyzw``, ``q_offset_matrix``, or composed with a
``wrist_tweaks`` entry.

Convention rules (must match anything that consumes a
``RobotRetargetConfig`` value):

* Matrices are row-major 3x3 lists.
* Quaternions are ``xyzw``.
* Distances are meters.
* Robot world convention: X = forward, Y = left, Z = up.
* Rotation composition is right-multiply:
  ``target_rot = R_world @ source_rot @ correction``.
* ``joint_offsets`` represent source-joint-local -> robot-link-local basis
  changes (right-multiplied onto the SOMA bone's global rotation).
* ``wrist_tweaks`` are post-correction right-multiplies on individual
  robot frames; rename to ``per_link_tweaks`` if you need them on
  non-wrist links.

Schema is intentionally minimal. Bump ``schema_version`` for changes
that are not backward-compatible (renamed/removed fields, or fields
whose defaults would silently alter existing IK behaviour). Optional
fields whose missing-value default is a strict no-op (e.g.
``posture_task`` defaulting to all-zero costs) do not need a bump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget import ASSETS_DIR

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
"""Default location for ``configs/<robot>/{frame_alignment,retargeter}.json``."""

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IkMapEntry:
    """One row of the IK end-effector map."""

    soma_joint: str
    position_cost: float
    orientation_cost: float


@dataclass(frozen=True)
class PostureTaskConfig:
    """Posture-regularization weights for the per-frame IK QP.

    Both costs default to ``0.0`` which means "no posture term at all"
    -- the JSON block is optional, and configs that omit it land at
    these defaults so existing behaviour is preserved bit-for-bit.

    The two costs are wired into Pink ``PostureTask`` instances that
    only act on the actuated DoFs (Pink strips the floating-base
    tangent prefix automatically). They are independent: ``q0_cost``
    pulls every frame's ``q`` toward ``robot.q0`` (a stationary
    regularizer), while ``q_prev_cost`` pulls toward the previous
    frame's IK solution (a temporal smoother).

    Attributes:
        q0_cost: Cost weight (``[homogeneous_cost] / rad``) for the
            "regularize toward ``robot.q0``" posture term. ``0.0``
            disables the term entirely.
        q_prev_cost: Cost weight for the "track previous-frame ``q``"
            posture term. On frame 0 this collapses to a pull toward
            ``q0`` (since the warm-start ``qpos`` is ``q0``).
        lm_damping: Levenberg-Marquardt damping used by both posture
            tasks. Same units as Pink's ``Task.lm_damping``.
        gain: Pink task gain in ``[0, 1]`` (low-pass-style filtering on
            the posture error). ``1.0`` is dead-beat tracking.
    """

    q0_cost: float = 0.0
    q_prev_cost: float = 0.0
    lm_damping: float = 0.0
    gain: float = 1.0


@dataclass(frozen=True)
class RobotRetargetConfig:
    """Final, fully materialized robot retarget config.

    Attributes:
        robot_name: Name of the robot (e.g. ``"g1"``).
        source_model: Source skeleton this config is wired against
            (e.g. ``"soma"``). The IK runtime uses this to dispatch to the
            correct loader; we don't model two source skeletons in the
            same config.
        urdf_path: Resolved absolute path to the URDF file.
        package_dirs: Resolved absolute paths used to resolve
            ``package://`` mesh URLs.
        ik_map: Mapping ``robot_frame -> IkMapEntry``.
        foot_frames: Robot frame names used for ground anchoring.
        ankle_roll_offset: Z distance from ``*_ankle_roll_link`` joint
            origin down to the foot sole (meters).
        base_source_joint: SOMA joint used as the scaling anchor; usually
            ``"Hips"``.
        r_world: ``(3, 3)`` rotation, source world -> robot world.
        r_per_bone: ``soma_joint -> (3, 3)`` correction. Right-multiplied
            onto the SOMA bone's global rotation; not currently consumed
            directly by the runtime, but kept so probes / migration
            scripts can inspect the per-bone offset before any per-link
            tweak is applied.
        r_per_link: ``robot_frame -> (3, 3)`` final composed correction
            returned by ``get_frame_rotation_correction``. This is what
            ``set_frame_tasks_target`` reads.
        t_per_link: ``robot_frame -> (3,)`` translation offset, expressed
            in the corrected robot-link local frame. Applied at runtime
            as ``target_pos += target_rot @ t_offset`` in
            ``set_frame_tasks_target`` (mirrors soma-retargeter's
            ``wp.quat_rotate(q, offset_tx.p)`` term in
            ``wp_compute_scaled_effectors``). Frames that have no
            mapping return zeros.
        joint_translation_offsets: ``soma_joint -> (3,)`` offset vector
            from the frame_alignment config; the runtime consumes the
            ``robot_frame``-keyed projection ``t_per_link`` instead, but
            this dict is kept for tools that need to look up the offset
            by SOMA joint.
        auto_derived_joints: SOMA joint names whose ``q_offset`` was
            computed by ``scripts/retarget/derive_soma_g1_corrections.py``
            (i.e. via the formula ``(R_world @ soma_world_rest).T @
            robot_world_q0``). The regression test uses this list to
            decide which rows must satisfy the q0-residual invariant;
            joints whose corrections were tuned empirically are
            intentionally absent from this list.
        posture_task: Optional posture-regularization weights. Defaults
            to all-zero costs (no posture term), which is a strict
            no-op compatible with the existing SOMA bit-equivalence
            regression. Consumed by
            ``ConfigDrivenWholeBodyKinematics``.
    """

    robot_name: str
    source_model: str
    urdf_path: Path
    package_dirs: list[Path]
    ik_map: dict[str, IkMapEntry]
    foot_frames: list[str]
    ankle_roll_offset: float
    base_source_joint: str
    r_world: np.ndarray
    r_per_bone: dict[str, np.ndarray]
    r_per_link: dict[str, np.ndarray]
    t_per_link: dict[str, np.ndarray] = field(default_factory=dict)
    joint_translation_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    auto_derived_joints: list[str] = field(default_factory=list)
    posture_task: PostureTaskConfig = field(default_factory=PostureTaskConfig)


def _decode_rotation(
    payload: dict[str, Any],
    *,
    matrix_key: str,
    quat_key: str,
    context: str,
) -> np.ndarray:
    """Decode either a 3x3 matrix field or an xyzw quaternion field.

    ``payload`` is the surrounding object, ``matrix_key`` and ``quat_key``
    name the two mutually-exclusive fields. Raises ``ValueError`` if both
    or neither are present.
    """
    has_matrix = matrix_key in payload
    has_quat = quat_key in payload
    if has_matrix == has_quat:
        raise ValueError(
            f"{context}: must specify exactly one of {matrix_key!r} or "
            f"{quat_key!r}; got matrix={has_matrix}, quat={has_quat}."
        )
    if has_matrix:
        mat = np.asarray(payload[matrix_key], dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(
                f"{context}: {matrix_key!r} must be a 3x3 list, got shape {mat.shape}."
            )
        return mat
    quat_xyzw = np.asarray(payload[quat_key], dtype=np.float64)
    if quat_xyzw.shape != (4,):
        raise ValueError(
            f"{context}: {quat_key!r} must be a 4-element xyzw quaternion, "
            f"got shape {quat_xyzw.shape}."
        )
    return R.from_quat(quat_xyzw).as_matrix()


def _check_schema_version(payload: dict[str, Any], path: Path) -> None:
    """Reject configs from a future or missing schema."""
    version = payload.get("schema_version")
    if version is None:
        raise ValueError(
            f"{path}: missing required field 'schema_version'. "
            f"Expected {SCHEMA_VERSION}."
        )
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {version} is not supported by this "
            f"loader (expected {SCHEMA_VERSION}). Update the loader or "
            f"regenerate the config."
        )


def _check_required_fields(
    payload: dict[str, Any], required: list[str], path: Path
) -> None:
    """Raise ``ValueError`` if any of ``required`` is missing."""
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"{path}: missing required fields: {missing}.")


def _parse_posture_task(payload: dict[str, Any], path: Path) -> PostureTaskConfig:
    """Parse the optional ``posture_task`` block from a retargeter payload.

    Missing block -> all-zero defaults, which the IK runtime treats as
    "no posture term"; this is the strict no-op path for backward
    compatibility with configs predating this field.
    """
    raw = payload.get("posture_task")
    if raw is None:
        return PostureTaskConfig()
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: 'posture_task' must be an object, got " f"{type(raw).__name__}."
        )
    allowed = {"q0_cost", "q_prev_cost", "lm_damping", "gain"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{path}: 'posture_task' contains unknown fields {sorted(unknown)}; "
            f"expected a subset of {sorted(allowed)}."
        )
    q0_cost = float(raw.get("q0_cost", 0.0))
    q_prev_cost = float(raw.get("q_prev_cost", 0.0))
    lm_damping = float(raw.get("lm_damping", 0.0))
    gain = float(raw.get("gain", 1.0))
    if q0_cost < 0.0 or q_prev_cost < 0.0:
        raise ValueError(
            f"{path}: 'posture_task' costs must be non-negative; got "
            f"q0_cost={q0_cost}, q_prev_cost={q_prev_cost}."
        )
    if lm_damping < 0.0:
        raise ValueError(
            f"{path}: 'posture_task.lm_damping' must be non-negative; got "
            f"{lm_damping}."
        )
    if not 0.0 <= gain <= 1.0:
        raise ValueError(f"{path}: 'posture_task.gain' must be in [0, 1]; got {gain}.")
    return PostureTaskConfig(
        q0_cost=q0_cost,
        q_prev_cost=q_prev_cost,
        lm_damping=lm_damping,
        gain=gain,
    )


def _load_frame_alignment(alignment_path: Path) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    str,
    str,
    list[str],
]:
    """Read ``frame_alignment.json``.

    Returns (R_world, r_per_bone, t_offsets, robot_name, source_model,
    auto_derived_joints).
    """
    with alignment_path.open() as f:
        payload = json.load(f)

    _check_schema_version(payload, alignment_path)
    _check_required_fields(
        payload,
        [
            "robot_name",
            "source_model",
            "joint_offsets",
        ],
        alignment_path,
    )

    robot_name = str(payload["robot_name"])
    source_model = str(payload["source_model"])

    r_world = _decode_rotation(
        payload,
        matrix_key="world_axis_swap_matrix",
        quat_key="world_axis_swap_xyzw",
        context=f"{alignment_path}: world axis swap",
    )

    r_per_bone: dict[str, np.ndarray] = {}
    t_offsets: dict[str, np.ndarray] = {}
    for soma_joint, entry in payload["joint_offsets"].items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{alignment_path}: joint_offsets[{soma_joint!r}] must be an "
                f"object, got {type(entry).__name__}."
            )
        r_per_bone[soma_joint] = _decode_rotation(
            entry,
            matrix_key="q_offset_matrix",
            quat_key="q_offset_xyzw",
            context=f"{alignment_path}: joint_offsets[{soma_joint!r}]",
        )
        t = entry.get("t_offset", [0.0, 0.0, 0.0])
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.shape != (3,):
            raise ValueError(
                f"{alignment_path}: joint_offsets[{soma_joint!r}].t_offset "
                f"must be a 3-element vector, got shape {t_arr.shape}."
            )
        t_offsets[soma_joint] = t_arr

    auto_derived_joints = [
        str(name) for name in payload.get("_auto_derived_joints", [])
    ]
    return (
        r_world,
        r_per_bone,
        t_offsets,
        robot_name,
        source_model,
        auto_derived_joints,
    )


def _load_retargeter(
    retargeter_path: Path,
    *,
    robot_name: str,
    source_model: str,
) -> tuple[
    Path,
    list[Path],
    dict[str, IkMapEntry],
    list[str],
    float,
    str,
    dict[str, np.ndarray],
    PostureTaskConfig,
]:
    """Read ``retargeter.json``; cross-check robot_name/source_model with frame_alignment."""
    with retargeter_path.open() as f:
        payload = json.load(f)

    _check_schema_version(payload, retargeter_path)
    _check_required_fields(
        payload,
        [
            "robot_name",
            "source_model",
            "urdf",
            "package_dirs",
            "ik_map",
            "foot_frames",
            "ankle_roll_offset",
            "base_source_joint",
        ],
        retargeter_path,
    )

    if str(payload["robot_name"]) != robot_name:
        raise ValueError(
            f"{retargeter_path}: robot_name mismatch; frame_alignment.json reports "
            f"{robot_name!r}, retargeter.json reports {payload['robot_name']!r}."
        )
    if str(payload["source_model"]) != source_model:
        raise ValueError(
            f"{retargeter_path}: source_model mismatch; frame_alignment.json reports "
            f"{source_model!r}, retargeter.json reports "
            f"{payload['source_model']!r}."
        )

    config_root = retargeter_path.parent
    urdf_path = (config_root / payload["urdf"]).resolve()
    package_dirs = [(config_root / p).resolve() for p in payload["package_dirs"]]

    ik_map_raw = payload["ik_map"]
    ik_map: dict[str, IkMapEntry] = {}
    for robot_frame, entry in ik_map_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{retargeter_path}: ik_map[{robot_frame!r}] must be an "
                f"object, got {type(entry).__name__}."
            )
        ik_map[robot_frame] = IkMapEntry(
            soma_joint=str(entry["soma_joint"]),
            position_cost=float(entry["position_cost"]),
            orientation_cost=float(entry["orientation_cost"]),
        )

    foot_frames = [str(name) for name in payload["foot_frames"]]
    ankle_roll_offset = float(payload["ankle_roll_offset"])
    base_source_joint = str(payload["base_source_joint"])

    per_link_tweaks: dict[str, np.ndarray] = {}
    raw_tweaks = payload.get("wrist_tweaks") or payload.get("per_link_tweaks") or {}
    for robot_frame, entry in raw_tweaks.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{retargeter_path}: per_link_tweaks[{robot_frame!r}] must "
                f"be an object, got {type(entry).__name__}."
            )
        per_link_tweaks[robot_frame] = _decode_rotation(
            entry,
            matrix_key="r_offset_matrix",
            quat_key="r_offset_xyzw",
            context=f"{retargeter_path}: per_link_tweaks[{robot_frame!r}]",
        )

    posture_task = _parse_posture_task(payload, retargeter_path)

    return (
        urdf_path,
        package_dirs,
        ik_map,
        foot_frames,
        ankle_roll_offset,
        base_source_joint,
        per_link_tweaks,
        posture_task,
    )


def _compose_per_link(
    *,
    ik_map: dict[str, IkMapEntry],
    r_per_bone: dict[str, np.ndarray],
    per_link_tweaks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build the final ``robot_frame -> 3x3`` correction the IK consumes.

    Composition: ``r_per_bone[ik_map[frame].soma_joint] @ per_link_tweaks[frame]``.
    Either piece may be missing; missing => identity.
    """
    out: dict[str, np.ndarray] = {}
    for robot_frame, entry in ik_map.items():
        bone_correction = r_per_bone.get(entry.soma_joint, np.eye(3))
        tweak = per_link_tweaks.get(robot_frame, np.eye(3))
        out[robot_frame] = bone_correction @ tweak
    return out


def load_robot_config(
    robot_name: str,
    *,
    configs_dir: Path | None = None,
) -> RobotRetargetConfig:
    """Load and validate the per-robot retarget config bundle.

    Args:
        robot_name: Folder name under ``configs_dir``.
        configs_dir: Override path to the configs directory; defaults to
            the in-repo ``configs/`` next to this module.

    Returns:
        A fully materialized :class:`RobotRetargetConfig`. All matrices
        are pre-composed numpy arrays; runtime code does not need to know
        whether the JSON used quaternions or matrices.
    """
    base = Path(configs_dir).resolve() if configs_dir is not None else CONFIGS_DIR
    robot_root = base / robot_name
    if not robot_root.is_dir():
        raise FileNotFoundError(
            f"Robot config directory not found: {robot_root}. Available "
            f"robots: {sorted(p.name for p in base.iterdir() if p.is_dir())}"
            if base.is_dir()
            else f"Robot config directory not found: {robot_root}."
        )

    alignment_path = robot_root / "frame_alignment.json"
    retargeter_path = robot_root / "retargeter.json"
    if not alignment_path.is_file():
        raise FileNotFoundError(f"Missing required config: {alignment_path}")
    if not retargeter_path.is_file():
        raise FileNotFoundError(f"Missing required config: {retargeter_path}")

    (
        r_world,
        r_per_bone,
        t_offsets,
        alignment_robot_name,
        source_model,
        auto_derived_joints,
    ) = _load_frame_alignment(alignment_path)

    if alignment_robot_name != robot_name:
        raise ValueError(
            f"{alignment_path}: robot_name field {alignment_robot_name!r} does "
            f"not match the requested {robot_name!r}."
        )

    (
        urdf_path,
        package_dirs,
        ik_map,
        foot_frames,
        ankle_roll_offset,
        base_source_joint,
        per_link_tweaks,
        posture_task,
    ) = _load_retargeter(
        retargeter_path,
        robot_name=robot_name,
        source_model=source_model,
    )

    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"{retargeter_path}: urdf path resolves to {urdf_path}, which does not exist."
        )

    r_per_link = _compose_per_link(
        ik_map=ik_map,
        r_per_bone=r_per_bone,
        per_link_tweaks=per_link_tweaks,
    )

    t_per_link: dict[str, np.ndarray] = {}
    for robot_frame, entry in ik_map.items():
        t_per_link[robot_frame] = t_offsets.get(
            entry.soma_joint, np.zeros(3, dtype=np.float64)
        )

    return RobotRetargetConfig(
        robot_name=robot_name,
        source_model=source_model,
        urdf_path=urdf_path,
        package_dirs=package_dirs,
        ik_map=ik_map,
        foot_frames=foot_frames,
        ankle_roll_offset=ankle_roll_offset,
        base_source_joint=base_source_joint,
        r_world=r_world,
        r_per_bone=r_per_bone,
        r_per_link=r_per_link,
        t_per_link=t_per_link,
        joint_translation_offsets=t_offsets,
        auto_derived_joints=auto_derived_joints,
        posture_task=posture_task,
    )


__all__ = [
    "CONFIGS_DIR",
    "SCHEMA_VERSION",
    "IkMapEntry",
    "PostureTaskConfig",
    "RobotRetargetConfig",
    "load_robot_config",
]


# Silence unused-import linters; ``ASSETS_DIR`` is intentionally re-exposed
# for tools that want to resolve mesh paths relative to the repo root.
_ = ASSETS_DIR
