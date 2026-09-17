# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free embodiment contracts for GR00T post-training workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FieldSpec:
    """One ordered modality field sourced from a named tensor slice."""

    key: str
    source_term: str
    start: int
    end: int
    transform: str | None = None

    def __post_init__(self) -> None:
        """Validate the serialized slice contract."""
        if not self.key or not self.source_term:
            raise ValueError("field key and source_term must be non-empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"invalid slice for {self.key!r}: [{self.start}:{self.end}]"
            )
        if self.transform not in {
            None,
            "quaternion_right_ref",
            "quaternion_left_ref",
        }:
            raise ValueError(f"unsupported field transform: {self.transform!r}")

    @property
    def width(self) -> int:
        """Return this field's flattened width."""
        return self.end - self.start


@dataclass(frozen=True)
class CameraSpec:
    """One required camera observation and its GR00T video key."""

    key: str
    observation_term: str
    sensor_name: str

    def __post_init__(self) -> None:
        """Require a complete explicit camera mapping."""
        if any(
            not value.strip()
            for value in (self.key, self.observation_term, self.sensor_name)
        ):
            raise ValueError(
                "camera key, observation_term, and sensor_name must be non-empty"
            )


@dataclass(frozen=True)
class EmbodimentContract:
    """Complete post-training data and inference contract for one embodiment."""

    contract_id: str
    robot_name: str
    fps: int
    action_horizon: int
    source_task: str
    record_task: str
    inference_task: str
    modality_config: str
    source_action_terms: tuple[str, ...]
    joint_names: tuple[str, ...]
    state_fields: tuple[FieldSpec, ...]
    action_fields: tuple[FieldSpec, ...]
    cameras: tuple[CameraSpec, ...]
    reset_joint_groups: Mapping[str, tuple[str, ...]]
    source_terminations: tuple[str, ...]
    evaluation_terminations: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate cross-field embodiment invariants."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version!r}; expected {SCHEMA_VERSION}"
            )
        for name, value in (
            ("contract_id", self.contract_id),
            ("robot_name", self.robot_name),
            ("source_task", self.source_task),
            ("record_task", self.record_task),
            ("inference_task", self.inference_task),
            ("modality_config", self.modality_config),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if any(not name.strip() for name in self.joint_names) or len(
            set(self.joint_names)
        ) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        for label, values in (
            ("state field keys", tuple(field.key for field in self.state_fields)),
            ("action field keys", tuple(field.key for field in self.action_fields)),
            ("camera keys", tuple(camera.key for camera in self.cameras)),
            (
                "camera observation terms",
                tuple(camera.observation_term for camera in self.cameras),
            ),
            (
                "camera sensor names",
                tuple(camera.sensor_name for camera in self.cameras),
            ),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{label} must be non-empty and unique")
        if (
            not self.source_action_terms
            or any(not name.strip() for name in self.source_action_terms)
            or len(set(self.source_action_terms)) != len(self.source_action_terms)
        ):
            raise ValueError("source_action_terms must be non-empty and unique")
        for label, names in (
            ("source_terminations", self.source_terminations),
            ("evaluation_terminations", self.evaluation_terminations),
        ):
            if (
                not names
                or any(not name.strip() for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"{label} must be non-empty and unique")
        action_width = sum(field.width for field in self.action_fields)
        if self.reset_joint_groups and not self.joint_names:
            raise ValueError(
                "reset joint groups require an explicit joint_names layout"
            )
        if self.joint_names and action_width != len(self.joint_names):
            raise ValueError(
                f"action field width must match joint_names: {action_width} != {len(self.joint_names)}"
            )
        if any(not name.strip() for name in self.reset_joint_groups):
            raise ValueError("reset joint group names must be non-empty")
        group_joints = [
            name for names in self.reset_joint_groups.values() for name in names
        ]
        if len(set(group_joints)) != len(group_joints):
            raise ValueError("reset joint groups must not overlap")
        unknown = sorted(set(group_joints) - set(self.joint_names))
        if unknown:
            raise ValueError(f"reset joint groups contain unknown joints: {unknown}")
        if self.joint_names and set(group_joints) != set(self.joint_names):
            missing = sorted(set(self.joint_names) - set(group_joints))
            raise ValueError(
                f"reset joint groups do not cover contract joints: {missing}"
            )

    @property
    def state_dim(self) -> int:
        """Return the flattened policy-state dimension."""
        return sum(field.width for field in self.state_fields)

    @property
    def action_dim(self) -> int:
        """Return the flattened policy-action dimension."""
        return sum(field.width for field in self.action_fields)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""
        value = asdict(self)
        value["reset_joint_groups"] = {
            name: list(joints) for name, joints in self.reset_joint_groups.items()
        }
        return value

    @property
    def sha256(self) -> str:
        """Return a stable semantic hash of the canonical representation."""
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _field(value: Mapping[str, Any]) -> FieldSpec:
    return FieldSpec(
        key=str(value["key"]),
        source_term=str(value["source_term"]),
        start=int(value["start"]),
        end=int(value["end"]),
        transform=(
            str(value["transform"]) if value.get("transform") is not None else None
        ),
    )


def _camera(value: Mapping[str, Any]) -> CameraSpec:
    return CameraSpec(
        key=str(value["key"]),
        observation_term=str(value["observation_term"]),
        sensor_name=str(value["sensor_name"]),
    )


def embodiment_contract_from_dict(value: Mapping[str, Any]) -> EmbodimentContract:
    """Validate and construct an embodiment contract from serialized data."""
    try:
        return EmbodimentContract(
            schema_version=int(value["schema_version"]),
            contract_id=str(value["contract_id"]),
            robot_name=str(value["robot_name"]),
            fps=int(value["fps"]),
            action_horizon=int(value["action_horizon"]),
            source_task=str(value["source_task"]),
            record_task=str(value["record_task"]),
            inference_task=str(value["inference_task"]),
            modality_config=str(value["modality_config"]),
            source_action_terms=tuple(
                str(name) for name in value["source_action_terms"]
            ),
            joint_names=tuple(str(name) for name in value["joint_names"]),
            state_fields=tuple(_field(field) for field in value["state_fields"]),
            action_fields=tuple(_field(field) for field in value["action_fields"]),
            cameras=tuple(_camera(camera) for camera in value["cameras"]),
            reset_joint_groups={
                str(name): tuple(str(joint) for joint in joints)
                for name, joints in value["reset_joint_groups"].items()
            },
            source_terminations=tuple(
                str(name) for name in value["source_terminations"]
            ),
            evaluation_terminations=tuple(
                str(name) for name in value["evaluation_terminations"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid embodiment contract: {exc}") from exc


_RIGHT_FINGERS = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)
VEGA_FINGER_JOINT_ORDER = _RIGHT_FINGERS + tuple(
    name.replace("right_", "left_", 1) for name in _RIGHT_FINGERS
)
VEGA_ARM_JOINT_ORDER = tuple(f"R_arm_j{i}" for i in range(1, 8)) + tuple(
    f"L_arm_j{i}" for i in range(1, 8)
)
VEGA_GROOT_JOINT_ORDER = (
    VEGA_ARM_JOINT_ORDER[:7]
    + VEGA_FINGER_JOINT_ORDER[:22]
    + VEGA_ARM_JOINT_ORDER[7:]
    + VEGA_FINGER_JOINT_ORDER[22:]
)

VEGA_SHARPA_JOINT = EmbodimentContract(
    contract_id="vega_sharpa_joint",
    robot_name="vega_sharpa",
    fps=20,
    action_horizon=16,
    source_task="VegaSharpa-WholeBody-Manip-v0",
    record_task="VegaSharpa-WholeBody-Gr00t-Record-v0",
    inference_task="VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0",
    modality_config="groot_finetune/vega_sharpa_joint_config.py",
    source_action_terms=("joint_pos",),
    joint_names=VEGA_GROOT_JOINT_ORDER,
    state_fields=(
        FieldSpec("right_arm", "arm_joint_pos", 0, 7),
        FieldSpec("left_arm", "arm_joint_pos", 7, 14),
        FieldSpec("right_finger", "finger_joint_pos", 0, 22),
        FieldSpec("left_finger", "finger_joint_pos", 22, 44),
    ),
    action_fields=(
        FieldSpec("right_arm", "action_target", 0, 7),
        FieldSpec("right_finger", "action_target", 7, 29),
        FieldSpec("left_arm", "action_target", 29, 36),
        FieldSpec("left_finger", "action_target", 36, 58),
    ),
    cameras=(
        CameraSpec("front", "image", "camera"),
        CameraSpec("right_wrist_view", "image_right_wrist", "camera_right_wrist"),
        CameraSpec("left_wrist_view", "image_left_wrist", "camera_left_wrist"),
    ),
    reset_joint_groups={
        "arm": VEGA_ARM_JOINT_ORDER,
        "finger": VEGA_FINGER_JOINT_ORDER,
    },
    source_terminations=("timeout",),
    evaluation_terminations=("timeout", "robot_state_diverged"),
)

SHARPA_DUAL_HAND_THREE_CAMERA = EmbodimentContract(
    contract_id="sharpa_dual_hand_three_camera",
    robot_name="sharpa",
    fps=20,
    action_horizon=16,
    source_task="Sharpa-V2D-v0",
    record_task="Sharpa-V2D-Gr00t-Record-v0",
    inference_task="Sharpa-V2D-Gr00t-Inference-v0",
    modality_config="groot_finetune/sharpa_dual_hand_three_camera_config.py",
    source_action_terms=(
        "right_joint_residual_action",
        "left_joint_residual_action",
    ),
    joint_names=(),
    state_fields=(
        FieldSpec("right_wrist_pos", "wrist_position_e", 0, 3),
        FieldSpec("left_wrist_pos", "wrist_position_e", 3, 6),
        FieldSpec(
            "right_wrist_quat",
            "wrist_orientation_e",
            0,
            4,
            "quaternion_right_ref",
        ),
        FieldSpec(
            "left_wrist_quat",
            "wrist_orientation_e",
            4,
            8,
            "quaternion_left_ref",
        ),
        FieldSpec("right_finger", "finger_joint_pos", 0, 22),
        FieldSpec("left_finger", "finger_joint_pos", 22, 44),
    ),
    action_fields=(
        FieldSpec("right_wrist_pos", "action_target", 0, 3),
        FieldSpec(
            "right_wrist_quat",
            "action_target",
            3,
            7,
            "quaternion_right_ref",
        ),
        FieldSpec("right_finger", "action_target", 7, 29),
        FieldSpec("left_wrist_pos", "action_target", 29, 32),
        FieldSpec(
            "left_wrist_quat",
            "action_target",
            32,
            36,
            "quaternion_left_ref",
        ),
        FieldSpec("left_finger", "action_target", 36, 58),
    ),
    cameras=(
        CameraSpec("front", "image", "camera"),
        CameraSpec("right_wrist_view", "image_right_wrist", "camera_right_wrist"),
        CameraSpec("left_wrist_view", "image_left_wrist", "camera_left_wrist"),
    ),
    reset_joint_groups={},
    source_terminations=("time_out",),
    evaluation_terminations=("time_out",),
)

_BUILTINS = {
    VEGA_SHARPA_JOINT.contract_id: VEGA_SHARPA_JOINT,
    SHARPA_DUAL_HAND_THREE_CAMERA.contract_id: SHARPA_DUAL_HAND_THREE_CAMERA,
}


def load_embodiment_contract(
    value: str | Path | Mapping[str, Any],
) -> EmbodimentContract:
    """Load a built-in ID, JSON file, or already-decoded contract."""
    if isinstance(value, Mapping):
        return embodiment_contract_from_dict(value)
    text = str(value)
    if text in _BUILTINS:
        return _BUILTINS[text]
    path = Path(value)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read embodiment contract {path}: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"embodiment contract must be a JSON object: {path}")
    return embodiment_contract_from_dict(decoded)


def write_embodiment_contract(path: Path, contract: EmbodimentContract) -> None:
    """Write a canonical contract snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
