# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dexmate Vega + SharpaWave hands embodiment (fixed base, two 7-DOF arms + Sharpa fingers).

Unlike the floating-hand SharpaHands, each wrist is positioned by a 7-DOF arm (``arm_dof_ids``),
not by floating prismatic+ball joints.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import newton
import numpy as np

from flash_chord.assets import ASSETS_DIR
from flash_chord.data.reference import NamedFrameReference, NamedRobotReference
from flash_chord.embodiments.base import (
    BodyFrame,
    Embodiment,
    EmbodimentLayout,
    HandLayout,
    HandLinkGeometry,
    JointControlSpec,
    ScalarJointLayout,
    register_embodiment,
    resolve_joint_controls,
)
from flash_chord.embodiments.binding import (
    RobotReferenceBinding,
    build_named_joint_trajectory,
    build_robot_reference_binding,
)
from flash_chord.embodiments.frames import capture_body_ids, resolve_body_frames
from flash_chord.embodiments.sharpa_collision import (
    capture_sharpa_link_geometry,
    filter_sharpa_collisions,
    sharpa_excluded_contacts,
    sharpa_object_contact_links,
)
from flash_chord.utils.labels import side_of

_VEGA_URDF = ASSETS_DIR / "urdfs" / "vega_sharpa" / "v2" / "vega_sharpa_58dof.urdf"
_VEGA_ASSET_ID = "vega_sharpa_v2"
_VEGA_ASSET_SHA256 = "88c2c061ad7ce13428e17c50e69ffdff08cbd5e8041654df33bbe7a7fb5e5bf2"

_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
_PALM_MOUNT_JOINTS = ("L_hand_mount", "R_hand_mount")


def _optional_sha256(value: str | None, label: str) -> str | None:
    if value is not None and (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hex digest or None, got {value!r}")
    return value


def _preserve_textured_mesh_materials(builder: newton.ModelBuilder, shape_start: int) -> None:
    """Prevent Newton's fallback palette from tinting imported mesh textures."""
    for shape_id in range(shape_start, len(builder.shape_source)):
        source = builder.shape_source[shape_id]
        if isinstance(source, newton.Mesh) and source.texture is not None and source.color is None:
            source.color = (1.0, 1.0, 1.0)
            builder.shape_color[shape_id] = source.color


def _joint_control_overrides() -> dict[str, dict]:
    """Calibrated Vega-arm and Sharpa-finger actuator parameters, by terminal joint name."""
    arm = (
        (48.68459466964289, 41.15447085114586, 0.55100, 100.0),
        (27.43237011630257, 23.159532675972144, 0.55100, 100.0),
        (29.62593291473207, 24.986205930444992, 0.19072, 80.0),
        (23.134317278891867, 19.532232049454098, 0.19072, 80.0),
        (2.5463846202007985, 2.148072759799482, 0.07232, 25.0),
        (4.227251607475805, 3.5765041461700604, 0.07232, 25.0),
        (3.4862647411242134, 2.9439891209118327, 0.07232, 25.0),
    )
    overrides = {
        rf"_arm_j{index}$": {
            "kp": kp,
            "kd": kd,
            "armature": armature,
            "effort_limit": effort,
            "velocity_limit": 2.4,
            "friction": None,
        }
        for index, (kp, kd, armature, effort) in enumerate(arm, start=1)
    }
    overrides.update(
        {
            r"_CMC_(?:FE|AA)$": {"armature": 0.0032, "effort_limit": 3.3, "friction": 0.132},
            r"_pinky_CMC$": {"armature": 0.00012, "effort_limit": 0.5285, "friction": 0.012},
            r"_MCP_(?:FE|AA)$": {"armature": 0.00265, "effort_limit": 1.864, "friction": 0.07456},
            r"_(?:thumb_)?IP$": {"armature": 0.0006, "effort_limit": 0.638, "friction": 0.01276},
            r"_PIP$": {"armature": 0.0006, "effort_limit": 0.638, "friction": 0.01276},
            r"_DIP$": {"armature": 0.00042, "effort_limit": 0.18937, "friction": 0.00378738},
        }
    )
    return overrides


@dataclass
class VegaSharpaConfig:
    urdf_path: Path = _VEGA_URDF
    asset_id: str | None = _VEGA_ASSET_ID
    asset_sha256: str | None = _VEGA_ASSET_SHA256
    gravity_compensation: bool = True
    arm_feedback_scale: float = 1.0
    default_joint: JointControlSpec = field(
        default_factory=lambda: JointControlSpec(
            kp=1.74533,
            kd=0.01745,
            armature=0.0006,
            effort_limit=0.638,
            velocity_limit=11.62,
            friction=0.01276,
        )
    )
    joint_overrides: dict[str, dict] = field(default_factory=_joint_control_overrides)
    preserve_palm_bodies: bool = True
    root_pose_tolerance: float = 1.0e-5
    joint_limit_tolerance: float = 1.0e-5
    validate_named_frames: bool = True
    frame_position_tolerance: float = 5.0e-4
    frame_orientation_tolerance: float = 1.0e-4
    dp_frame_orientation_tolerance: float = 1.0e-3
    interpolated_frame_position_tolerance: float | None = 0.02
    interpolated_frame_orientation_tolerance: float | None = 0.1
    semantic_frame_names: tuple[str, ...] = ("zed_depth_frame", "zed_left_camera", "zed_right_camera")
    exclude_contacts: list[tuple[str, str]] = field(default_factory=sharpa_excluded_contacts)
    object_contact_links: list[str] = field(default_factory=sharpa_object_contact_links)

    def __post_init__(self) -> None:
        self.urdf_path = Path(self.urdf_path)
        self.semantic_frame_names = tuple(self.semantic_frame_names)
        if not np.isfinite(self.arm_feedback_scale) or self.arm_feedback_scale <= 0.0:
            raise ValueError(f"arm feedback scale must be finite and positive, got {self.arm_feedback_scale}")
        if self.asset_id is not None and (not isinstance(self.asset_id, str) or not self.asset_id):
            raise ValueError(f"asset ID must be a nonempty string or None, got {self.asset_id!r}")
        self.asset_sha256 = _optional_sha256(self.asset_sha256, "asset SHA-256")
        if (self.asset_id is None) != (self.asset_sha256 is None):
            raise ValueError("asset ID and asset SHA-256 must either both be configured or both be omitted")
        if len(set(self.semantic_frame_names)) != len(self.semantic_frame_names) or any(
            not isinstance(name, str) or not name for name in self.semantic_frame_names
        ):
            raise ValueError(f"semantic frame names must be unique and nonempty, got {self.semantic_frame_names}")


@register_embodiment("dexmate_sharpa")
class VegaSharpa:
    """Dexmate Vega torso + two 7-DOF arms + SharpaWave hands (fixed base)."""

    name = "dexmate_sharpa"

    def __init__(self, config: VegaSharpaConfig | None = None) -> None:
        self.config = config or VegaSharpaConfig()

    def build(self, builder: newton.ModelBuilder) -> EmbodimentLayout:
        self._validate_asset()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        shape_start = len(builder.shape_source)
        builder.add_urdf(
            source=str(self.config.urdf_path),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=True,
            ignore_inertial_definitions=False,
            force_position_velocity_actuation=True,
        )
        _preserve_textured_mesh_materials(builder, shape_start)
        semantic_body_ids = capture_body_ids(builder, _semantic_frame_names(self.config.semantic_frame_names))
        palm_body_ids = {side: semantic_body_ids[f"{side}_hand_C_MC"] for side in ("left", "right")}
        hand_geometry = capture_sharpa_link_geometry(
            builder,
            palm_body_ids=palm_body_ids,
            sides=("left", "right"),
            object_contact_links=self.config.object_contact_links,
        )
        filter_sharpa_collisions(
            builder,
            hand_geometry,
            protected_fixed_bodies=palm_body_ids.values(),
            excluded_contacts=self.config.exclude_contacts,
        )
        shape_count_before_collapse = len(builder.shape_body)
        joints_to_keep = []
        if self.config.preserve_palm_bodies:
            joints_to_keep.extend(_terminal_joint_labels(builder, _PALM_MOUNT_JOINTS))
        additional_body_ids = tuple(semantic_body_ids[name] for name in self.config.semantic_frame_names)
        joints_to_keep.extend(_frame_anchor_joint_labels(builder, additional_body_ids))
        collapse_result = builder.collapse_fixed_joints(joints_to_keep=joints_to_keep or None)
        if len(builder.shape_body) != shape_count_before_collapse:
            raise ValueError("fixed-joint collapse changed shape IDs required by hand geometry ownership")
        semantic_frames = {frame.name: frame for frame in resolve_body_frames(semantic_body_ids, collapse_result)}
        self._apply_control_config(builder)
        return self._extract_layout(builder, semantic_frames, hand_geometry)

    def _validate_asset(self) -> None:
        path = self.config.urdf_path
        if not path.is_file():
            raise FileNotFoundError(f"Vega Sharpa URDF does not exist: {path}")
        expected = self.config.asset_sha256
        if expected is None:
            return
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            label = self.config.asset_id or str(path)
            raise ValueError(f"Vega Sharpa asset {label!r} SHA-256 mismatch: expected {expected}, got {digest}")

    def _apply_control_config(self, builder: newton.ModelBuilder) -> None:
        cfg = self.config
        joint_ids = tuple(
            joint for joint, joint_type in enumerate(builder.joint_type) if joint_type != newton.JointType.FIXED
        )
        specs = resolve_joint_controls(
            tuple(builder.joint_label[joint] for joint in joint_ids),
            cfg.default_joint,
            cfg.joint_overrides,
        )
        for j, spec in zip(joint_ids, specs, strict=True):
            label = builder.joint_label[j]
            feedback_scale = cfg.arm_feedback_scale if re.search(r"(?:^|/)(?:L|R)_arm_j[1-7]$", label) else 1.0
            qd_start = builder.joint_qd_start[j]
            q_start = builder.joint_q_start[j]
            lin, ang = builder.joint_dof_dim[j]
            for k in range(lin + ang):
                dof = qd_start + k
                builder.joint_target_ke[dof] = feedback_scale * spec.kp
                builder.joint_target_kd[dof] = feedback_scale * spec.kd
                builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
                builder.joint_target_pos[dof] = spec.default_pos
                builder.joint_armature[dof] = spec.armature
                builder.joint_effort_limit[dof] = spec.effort_limit
                if spec.velocity_limit is not None:
                    builder.joint_velocity_limit[dof] = spec.velocity_limit
                if spec.friction is not None:
                    builder.joint_friction[dof] = spec.friction
                builder.joint_q[q_start + k] = spec.default_pos
        if cfg.gravity_compensation:
            actgravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"].values
            for dof in range(builder.joint_dof_count):
                actgravcomp[dof] = True
            body_gravcomp = builder.custom_attributes["mujoco:gravcomp"].values
            for b in range(len(builder.body_label)):
                body_gravcomp[b] = 1.0

    def _extract_layout(
        self,
        builder: newton.ModelBuilder,
        semantic_frames: dict[str, BodyFrame],
        hand_geometry: dict[str, tuple[HandLinkGeometry, ...]],
    ) -> EmbodimentLayout:
        joints: dict[str, dict[str, list[tuple[int, int, str]]]] = {
            side: {"arm": [], "finger": []} for side in ("left", "right")
        }
        for j, label in enumerate(builder.joint_label):
            if builder.joint_type[j] == newton.JointType.FIXED:
                continue
            side = side_of(label, unknown=None)
            if side is None:
                raise ValueError(f"movable joint cannot be assigned to a hand: {label!r}")
            qd_start = builder.joint_qd_start[j]
            q_start = builder.joint_q_start[j]
            lin, ang = builder.joint_dof_dim[j]
            if builder.joint_type[j] != newton.JointType.REVOLUTE or lin + ang != 1:
                raise ValueError(f"Vega Sharpa expects scalar revolute joints, got {label!r}")
            name = label.rsplit("/", 1)[-1]
            group = "arm" if "_arm_j" in name else "finger"
            joints[side][group].append((qd_start, q_start, name))

        hands: list[HandLayout] = []
        for side in ("left", "right"):
            arm = sorted(joints[side]["arm"])
            finger = sorted(joints[side]["finger"])
            if len(arm) != 7 or len(finger) != 22:
                raise ValueError(f"{side} Vega Sharpa layout must have 7 arm and 22 finger joints")
            palm = semantic_frames[f"{side}_hand_C_MC"]
            dp_frames = tuple(semantic_frames[f"{side}_{digit}_DP"] for digit in _DIGITS)
            fingertip_frames = tuple(semantic_frames[f"{side}_{digit}_fingertip"] for digit in _DIGITS)
            hands.append(
                HandLayout(
                    side=side,
                    arm_dof_ids=tuple(dof for dof, _, _ in arm),
                    arm_q_ids=tuple(q_id for _, q_id, _ in arm),
                    arm_joint_names=tuple(name for _, _, name in arm),
                    finger_dof_ids=tuple(dof for dof, _, _ in finger),
                    link_geometry=hand_geometry[side],
                    finger_q_ids=tuple(q_id for _, q_id, _ in finger),
                    finger_joint_names=tuple(name for _, _, name in finger),
                    palm_frame=palm,
                    dp_frames=dp_frames,
                    fingertip_frames=fingertip_frames,
                )
            )
        scalar_joints = sorted(
            (
                (q_id, dof_id, name)
                for hand in hands
                for q_id, dof_id, name in zip(
                    hand.joint_q_ids,
                    hand.joint_dof_ids,
                    hand.joint_names,
                    strict=True,
                )
            ),
            key=lambda joint: joint[0],
        )
        layout = EmbodimentLayout(
            num_joint_q=len(builder.joint_q),
            num_joint_dof=builder.joint_dof_count,
            hands=tuple(hands),
            scalar_joints=ScalarJointLayout(
                q_ids=tuple(q_id for q_id, _, _ in scalar_joints),
                dof_ids=tuple(dof_id for _, dof_id, _ in scalar_joints),
                names=tuple(name for _, _, name in scalar_joints),
            ),
            semantic_frames=tuple(semantic_frames.values()),
        )
        q_ids = sorted(layout.scalar_joints.q_ids)
        dof_ids = sorted(layout.scalar_joints.dof_ids)
        if q_ids != list(range(layout.num_joint_q)) or dof_ids != list(range(layout.num_joint_dof)):
            raise ValueError("Vega Sharpa movable joint layout must cover every q coordinate and DOF exactly once")
        return layout

    def bind_reference(
        self,
        builder: newton.ModelBuilder,
        layout: EmbodimentLayout,
        reference,
    ) -> RobotReferenceBinding:
        """Bind a named whole-robot trajectory to the exact Vega joint and frame layout."""
        if not isinstance(reference, NamedRobotReference):
            raise TypeError("VegaSharpa requires a NamedRobotReference")
        source_names = tuple(reference.robot_joint_names())
        source_q = np.asarray(reference.robot_joint_pos(), dtype=np.float32)
        if source_q.ndim != 2 or source_q.shape[0] != reference.num_frames:
            raise ValueError(
                f"robot_joint_pos frame axis must have length {reference.num_frames}, got {source_q.shape}"
            )
        _validate_fixed_root(reference, self.config.root_pose_tolerance)
        joint_q, joint_target = build_named_joint_trajectory(
            builder,
            layout,
            source_names,
            source_q,
            joint_limit_tolerance=self.config.joint_limit_tolerance,
        )

        binding = build_robot_reference_binding(builder, layout, joint_q, joint_target, reference.fps)
        if self.config.validate_named_frames:
            _validate_named_frame_poses(binding, layout, reference, self.config)
        return binding


def _semantic_frame_names(additional: tuple[str, ...] = ()) -> tuple[str, ...]:
    hand_frames = tuple(
        name
        for side in ("left", "right")
        for name in (
            f"{side}_hand_C_MC",
            *(f"{side}_{digit}_DP" for digit in _DIGITS),
            *(f"{side}_{digit}_fingertip" for digit in _DIGITS),
        )
    )
    duplicate = set(hand_frames).intersection(additional)
    if duplicate:
        raise ValueError(f"additional semantic frames duplicate hand frames: {sorted(duplicate)}")
    return hand_frames + additional


def _terminal_joint_labels(builder: newton.ModelBuilder, names: tuple[str, ...]) -> list[str]:
    """Resolve terminal URDF joint names to Newton's full imported labels."""
    matches = {name: [] for name in names}
    for label in builder.joint_label:
        terminal = label.rsplit("/", 1)[-1]
        if terminal in matches:
            matches[terminal].append(label)
    invalid = {name: labels for name, labels in matches.items() if len(labels) != 1}
    if invalid:
        raise ValueError(f"unable to resolve fixed joints to preserve: {invalid}")
    return [matches[name][0] for name in names]


def _frame_anchor_joint_labels(
    builder: newton.ModelBuilder,
    body_ids: tuple[int, ...],
) -> list[str]:
    """Retain the nearest massive fixed ancestor for frames otherwise merged into the world."""
    incoming = {child: joint for joint, child in enumerate(builder.joint_child)}
    retained = []
    for requested_body in body_ids:
        body = requested_body
        while float(builder.body_mass[body]) <= 0.0:
            joint = incoming.get(body)
            if joint is None or builder.joint_parent[joint] < 0:
                raise ValueError(f"semantic body {builder.body_label[requested_body]!r} has no massive ancestor")
            body = builder.joint_parent[joint]
        joint = incoming.get(body)
        if joint is not None and builder.joint_type[joint] == newton.JointType.FIXED:
            retained.append(builder.joint_label[joint])
    return list(dict.fromkeys(retained))


def _duplicates(names: tuple[str, ...]) -> list[str]:
    return sorted({name for name in names if names.count(name) > 1})


def _validate_fixed_root(reference: NamedRobotReference, tolerance: float) -> None:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"root pose tolerance must be positive and finite, got {tolerance}")
    position = np.asarray(reference.robot_root_pos_w(), dtype=np.float32)
    quaternion = np.asarray(reference.robot_root_quat_w(), dtype=np.float32)
    if (
        position.shape != (reference.num_frames, 3)
        or quaternion.shape != (reference.num_frames, 4)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
    ):
        raise ValueError("fixed-base root trajectories have invalid shapes")
    quaternion_norm = np.linalg.norm(quaternion, axis=-1)
    if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("fixed-base root trajectory contains non-unit quaternions")
    quaternion = quaternion / quaternion_norm[:, None]
    position_error = np.linalg.norm(position, axis=-1)
    orientation_error = 2.0 * np.arccos(np.clip(np.abs(quaternion[:, 0]), 0.0, 1.0))
    if np.max(position_error) > tolerance or np.max(orientation_error) > tolerance:
        raise ValueError(
            "VegaSharpa is fixed-base but the reference root moves: "
            f"max_position={np.max(position_error):.6g}, max_orientation={np.max(orientation_error):.6g}"
        )


def _validate_named_frame_poses(
    binding: RobotReferenceBinding,
    layout: EmbodimentLayout,
    reference,
    config: VegaSharpaConfig,
) -> None:
    if not isinstance(reference, NamedFrameReference):
        raise TypeError("VegaSharpa frame validation requires a NamedFrameReference")
    source_names = tuple(reference.robot_frame_names())
    duplicate_source = _duplicates(source_names)
    if duplicate_source:
        raise ValueError(f"named frame reference has duplicate frames: {duplicate_source}")
    source_index = {name: index for index, name in enumerate(source_names)}
    palm_tip_names: list[str] = []
    palm_tip_position: list[np.ndarray] = []
    palm_tip_orientation: list[np.ndarray] = []
    dp_names: list[str] = []
    dp_position: list[np.ndarray] = []
    dp_orientation: list[np.ndarray] = []
    for side in layout.sides:
        hand_layout = layout.hand(side)
        hand = binding.hand(side)
        palm_tip_names.extend(frame.name for frame in (hand_layout.palm_frame, *hand_layout.fingertip_frames))
        palm_tip_position.append(np.concatenate((hand.wrist_pos_w[:, None], hand.fingertip_pos_w), axis=1))
        palm_tip_orientation.append(np.concatenate((hand.wrist_quat_w[:, None], hand.fingertip_quat_w), axis=1))
        dp_names.extend(frame.name for frame in hand_layout.dp_frames)
        dp_position.append(hand.dp_pos_w)
        dp_orientation.append(hand.dp_quat_w)
    if all(name in source_index for name in palm_tip_names):
        expected_names = palm_tip_names
        achieved_position = palm_tip_position
        achieved_orientation = palm_tip_orientation
        orientation_tolerance = config.frame_orientation_tolerance
    elif all(name in source_index for name in dp_names):
        expected_names = dp_names
        achieved_position = dp_position
        achieved_orientation = dp_orientation
        orientation_tolerance = config.dp_frame_orientation_tolerance
    else:
        missing_palm_tip = [name for name in palm_tip_names if name not in source_index]
        missing_dp = [name for name in dp_names if name not in source_index]
        raise ValueError(
            "named frame reference must provide either palm/fingertip or DP validation frames; "
            f"missing palm/fingertip={missing_palm_tip}, missing DP={missing_dp}"
        )
    source_position = np.asarray(reference.robot_frame_pos_w(), dtype=np.float32)
    source_orientation = np.asarray(reference.robot_frame_quat_w(), dtype=np.float32)
    if source_position.shape != (reference.num_frames, len(source_names), 3) or not np.all(
        np.isfinite(source_position)
    ):
        raise ValueError("robot_frame_pos_w shape does not match its frame names")
    if source_orientation.shape != (reference.num_frames, len(source_names), 4) or not np.all(
        np.isfinite(source_orientation)
    ):
        raise ValueError("robot_frame_quat_w shape does not match its frame names")
    source_orientation_norm = np.linalg.norm(source_orientation, axis=-1)
    if not np.allclose(source_orientation_norm, 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("robot_frame_quat_w contains non-unit quaternions")
    achieved_position_array = np.concatenate(achieved_position, axis=1)
    achieved_orientation_array = np.concatenate(achieved_orientation, axis=1)
    diagnostic_position = source_position[:, [source_index[name] for name in expected_names]]
    diagnostic_orientation = source_orientation[:, [source_index[name] for name in expected_names]]
    position_error = np.linalg.norm(achieved_position_array - diagnostic_position, axis=-1)
    orientation_error = _orientation_error(achieved_orientation_array, diagnostic_orientation)

    position_tolerance = _positive_tolerance(config.frame_position_tolerance, "frame position")
    orientation_tolerance = _positive_tolerance(orientation_tolerance, "frame orientation")
    strict_frames = np.arange(reference.num_frames) if not reference.metadata.is_resampled else np.array([0, -1])
    strict_kind = "native" if not reference.metadata.is_resampled else "source endpoint"
    _validate_frame_error(position_error, strict_frames, expected_names, position_tolerance, strict_kind, "m")
    _validate_frame_error(
        orientation_error,
        strict_frames,
        expected_names,
        orientation_tolerance,
        strict_kind,
        "rad",
    )

    if reference.metadata.is_resampled:
        if config.interpolated_frame_position_tolerance is not None:
            interpolation_position = _positive_tolerance(
                config.interpolated_frame_position_tolerance,
                "interpolated frame position",
            )
            _validate_frame_error(
                position_error,
                np.arange(reference.num_frames),
                expected_names,
                interpolation_position,
                "interpolated",
                "m",
            )
        if config.interpolated_frame_orientation_tolerance is not None:
            interpolation_orientation = _positive_tolerance(
                config.interpolated_frame_orientation_tolerance,
                "interpolated frame orientation",
            )
            _validate_frame_error(
                orientation_error,
                np.arange(reference.num_frames),
                expected_names,
                interpolation_orientation,
                "interpolated",
                "rad",
            )


def _orientation_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = np.linalg.norm(left, axis=-1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=-1, keepdims=True)
    if np.any(left_norm <= np.finfo(np.float32).eps) or np.any(right_norm <= np.finfo(np.float32).eps):
        raise ValueError("named frame orientations must not contain zero quaternions")
    left = left / left_norm
    right = right / right_norm
    dot = np.sum(left * right, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def _positive_tolerance(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} tolerance must be positive and finite, got {value}")
    return float(value)


def _validate_frame_error(
    error: np.ndarray,
    frame_ids: np.ndarray,
    names: list[str],
    tolerance: float,
    kind: str,
    unit: str,
) -> None:
    selected = error[frame_ids]
    if np.max(selected) <= tolerance:
        return
    selected_frame, point = np.unravel_index(np.argmax(selected), selected.shape)
    frame = int(frame_ids[selected_frame] % error.shape[0])
    raise ValueError(
        f"{kind} named frame {names[point]!r} disagrees with q-derived FK at frame {frame}: "
        f"{selected[selected_frame, point]:.6g} {unit} > {tolerance:.6g} {unit}"
    )


assert isinstance(VegaSharpa(), Embodiment)
