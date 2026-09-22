# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Floating-base Unitree G1 with two Dex3 hands."""

from __future__ import annotations

import hashlib
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
from flash_chord.embodiments.frames import capture_body_ids

_G1_DEX3_URDF = ASSETS_DIR / "urdfs" / "g1" / "main_with_hand.urdf"
_SIDES = ("left", "right")
_DIGITS = ("thumb", "middle", "index")
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0

_ASSET_ID = "g1_dex3_upstream_main"
_ASSET_SHA256 = "8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9"
_ASSET_CYLINDER_COLLISION_COUNT = 26

G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_DEX3_JOINT_NAMES = tuple(
    f"{side}_hand_{digit}_{index}_joint"
    for side in _SIDES
    for digit, count in (("thumb", 3), ("middle", 2), ("index", 2))
    for index in range(count)
)


def _implicit_motor(armature: float, effort_limit: float, velocity_limit: float, scale: float = 1.0) -> dict:
    armature *= scale
    return {
        "kp": armature * _NATURAL_FREQUENCY**2,
        "kd": 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQUENCY,
        "armature": armature,
        "effort_limit": effort_limit,
        "velocity_limit": velocity_limit,
        "friction": None,
    }


def _joint_control_overrides() -> dict[str, dict]:
    controls = {
        r"_(?:hip_pitch|hip_roll|knee)_joint$": _implicit_motor(_ARMATURE_7520_22, 139.0, 20.0),
        r"_hip_yaw_joint$": _implicit_motor(_ARMATURE_7520_14, 88.0, 32.0),
        r"_ankle_(?:pitch|roll)_joint$": _implicit_motor(_ARMATURE_5020, 50.0, 37.0, scale=2.0),
        r"waist_(?:roll|pitch)_joint$": _implicit_motor(_ARMATURE_5020, 50.0, 37.0, scale=2.0),
        r"waist_yaw_joint$": _implicit_motor(_ARMATURE_7520_14, 88.0, 32.0),
        r"_(?:shoulder_(?:pitch|roll|yaw)|elbow|wrist_roll)_joint$": _implicit_motor(_ARMATURE_5020, 25.0, 37.0),
        r"_wrist_(?:pitch|yaw)_joint$": _implicit_motor(_ARMATURE_4010, 5.0, 22.0),
        r"_hand_(?:thumb_[0-2]|middle_[0-1]|index_[0-1])_joint$": {
            "kp": 2.0,
            "kd": 0.2,
            "armature": 0.00149,
            "effort_limit": 0.76,
            "velocity_limit": 23.0,
            "friction": None,
        },
        r"_hip_pitch_joint$": {"default_pos": -0.312},
        r"_knee_joint$": {"default_pos": 0.669},
        r"_ankle_pitch_joint$": {"default_pos": -0.363},
        r"_elbow_joint$": {"default_pos": 0.6},
        r"left_shoulder_roll_joint$": {"default_pos": 0.2},
        r"left_shoulder_pitch_joint$": {"default_pos": 0.2},
        r"right_shoulder_roll_joint$": {"default_pos": -0.2},
        r"right_shoulder_pitch_joint$": {"default_pos": 0.2},
    }
    return controls


@dataclass
class G1Dex3Config:
    urdf_path: Path = _G1_DEX3_URDF
    asset_id: str | None = _ASSET_ID
    asset_sha256: str | None = _ASSET_SHA256
    replace_cylinders_with_capsules: bool = True
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.76)
    base_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    gravity_compensation: bool = False
    default_joint: JointControlSpec = field(
        default_factory=lambda: JointControlSpec(
            kp=0.0,
            kd=0.0,
            armature=0.0,
            effort_limit=1.0,
            velocity_limit=None,
            friction=None,
        )
    )
    joint_overrides: dict[str, dict] = field(default_factory=_joint_control_overrides)
    joint_limit_tolerance: float = 1.0e-5
    validate_palm_frames: bool = True
    frame_position_tolerance: float = 1.0e-4
    frame_orientation_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        self.urdf_path = Path(self.urdf_path)
        position = np.asarray(self.base_position, dtype=np.float64)
        quaternion = np.asarray(self.base_quat_xyzw, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"G1 base position must be a finite vec3, got {self.base_position}")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError(f"G1 base quaternion must be a finite xyzw value, got {self.base_quat_xyzw}")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-8:
            raise ValueError("G1 base quaternion must have nonzero norm")
        self.base_position = tuple(float(value) for value in position)
        self.base_quat_xyzw = tuple(float(value) for value in quaternion / norm)
        if self.asset_id is not None and (not isinstance(self.asset_id, str) or not self.asset_id):
            raise ValueError(f"asset ID must be a nonempty string or None, got {self.asset_id!r}")
        if self.asset_sha256 is not None and (
            not isinstance(self.asset_sha256, str)
            or len(self.asset_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.asset_sha256)
        ):
            raise ValueError(f"asset SHA-256 must be a lowercase hex digest or None, got {self.asset_sha256!r}")
        if (self.asset_id is None) != (self.asset_sha256 is None):
            raise ValueError("asset ID and asset SHA-256 must either both be configured or both be omitted")
        for name in ("joint_limit_tolerance", "frame_position_tolerance", "frame_orientation_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite, got {value}")


@register_embodiment("g1_dex3")
class G1Dex3:
    """Unitree G1 model-12 body with articulated Dex3 hands and a free pelvis."""

    name = "g1_dex3"

    def __init__(self, config: G1Dex3Config | None = None) -> None:
        self.config = config or G1Dex3Config()

    def build(self, builder: newton.ModelBuilder) -> EmbodimentLayout:
        import warp as wp

        self._validate_asset()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        shape_start = builder.shape_count
        builder.add_urdf(
            source=str(self.config.urdf_path),
            xform=wp.transform(wp.vec3(*self.config.base_position), wp.quat(*self.config.base_quat_xyzw)),
            floating=True,
            collapse_fixed_joints=False,
            enable_self_collisions=True,
            ignore_inertial_definitions=False,
            force_position_velocity_actuation=False,
        )
        if self.config.replace_cylinders_with_capsules:
            replaced = _replace_cylinders_with_capsules(builder, shape_start)
            if self.config.asset_sha256 == _ASSET_SHA256 and replaced != _ASSET_CYLINDER_COLLISION_COUNT:
                raise ValueError(
                    "pinned G1 Dex3 asset must contain "
                    f"{_ASSET_CYLINDER_COLLISION_COUNT} cylinder colliders, got {replaced}"
                )
        body_ids = capture_body_ids(builder, _semantic_body_names())
        hand_geometry = _capture_hand_geometry(builder, body_ids)
        frames = {name: BodyFrame(name=name, body_id=body_id) for name, body_id in body_ids.items()}
        self._apply_control_config(builder)
        return self._extract_layout(builder, frames, hand_geometry)

    def _validate_asset(self) -> None:
        path = self.config.urdf_path
        if not path.is_file():
            raise FileNotFoundError(f"G1 Dex3 URDF does not exist: {path}")
        expected = self.config.asset_sha256
        if expected is None:
            return
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"G1 Dex3 asset {self.config.asset_id!r} SHA-256 mismatch: expected {expected}, got {digest}"
            )

    def _apply_control_config(self, builder: newton.ModelBuilder) -> None:
        joint_ids = tuple(
            joint
            for joint, joint_type in enumerate(builder.joint_type)
            if joint_type not in (newton.JointType.FIXED, newton.JointType.FREE)
        )
        labels = tuple(builder.joint_label[joint] for joint in joint_ids)
        specs = resolve_joint_controls(labels, self.config.default_joint, self.config.joint_overrides)
        for joint, spec in zip(joint_ids, specs, strict=True):
            q_id = builder.joint_q_start[joint]
            dof_id = builder.joint_qd_start[joint]
            if builder.joint_type[joint] != newton.JointType.REVOLUTE or builder.joint_dof_dim[joint] != (0, 1):
                raise ValueError(f"G1 Dex3 expects scalar revolute joints, got {builder.joint_label[joint]!r}")
            builder.joint_target_ke[dof_id] = spec.kp
            builder.joint_target_kd[dof_id] = spec.kd
            builder.joint_target_mode[dof_id] = int(newton.JointTargetMode.POSITION)
            builder.joint_target_pos[dof_id] = spec.default_pos
            builder.joint_armature[dof_id] = spec.armature
            builder.joint_effort_limit[dof_id] = spec.effort_limit
            if spec.velocity_limit is not None:
                builder.joint_velocity_limit[dof_id] = spec.velocity_limit
            if spec.friction is not None:
                builder.joint_friction[dof_id] = spec.friction
            builder.joint_q[q_id] = spec.default_pos
        if self.config.gravity_compensation:
            builder.custom_attributes["mujoco:jnt_actgravcomp"].values[:] = [True] * builder.joint_dof_count
            builder.custom_attributes["mujoco:gravcomp"].values[:] = [1.0] * len(builder.body_label)

    def _extract_layout(
        self,
        builder: newton.ModelBuilder,
        frames: dict[str, BodyFrame],
        hand_geometry: dict[str, tuple[HandLinkGeometry, ...]],
    ) -> EmbodimentLayout:
        scalar: list[tuple[int, int, str]] = []
        hand_joints = {side: {"arm": [], "finger": []} for side in _SIDES}
        for joint, (label, joint_type) in enumerate(zip(builder.joint_label, builder.joint_type, strict=True)):
            if joint_type in (newton.JointType.FIXED, newton.JointType.FREE):
                continue
            if joint_type != newton.JointType.REVOLUTE or builder.joint_dof_dim[joint] != (0, 1):
                raise ValueError(f"G1 Dex3 expects scalar revolute joints, got {label!r}")
            name = label.rsplit("/", 1)[-1]
            entry = (builder.joint_qd_start[joint], builder.joint_q_start[joint], name)
            scalar.append((entry[1], entry[0], name))
            for side in _SIDES:
                if name in _arm_joint_names(side):
                    hand_joints[side]["arm"].append(entry)
                elif name in _finger_joint_names(side):
                    hand_joints[side]["finger"].append(entry)

        hands = []
        semantic_frames = [frames["pelvis"]]
        for side in _SIDES:
            arm = sorted(hand_joints[side]["arm"])
            finger = sorted(hand_joints[side]["finger"])
            if len(arm) != 7 or len(finger) != 7:
                raise ValueError(f"{side} G1 Dex3 layout must have 7 arm and 7 finger joints")
            palm = frames[f"{side}_hand_palm_link"]
            dp_frames = tuple(frames[f"{side}_hand_{digit}_{2 if digit == 'thumb' else 1}_link"] for digit in _DIGITS)
            fingertip_frames = tuple(_fingertip_frame(side, digit, frame) for digit, frame in zip(_DIGITS, dp_frames))
            hand = HandLayout(
                side=side,
                arm_dof_ids=tuple(dof for dof, _, _ in arm),
                arm_q_ids=tuple(q_id for _, q_id, _ in arm),
                arm_joint_names=tuple(name for _, _, name in arm),
                finger_dof_ids=tuple(dof for dof, _, _ in finger),
                finger_q_ids=tuple(q_id for _, q_id, _ in finger),
                finger_joint_names=tuple(name for _, _, name in finger),
                link_geometry=hand_geometry[side],
                palm_frame=palm,
                dp_frames=dp_frames,
                fingertip_frames=fingertip_frames,
            )
            hands.append(hand)
            semantic_frames.extend((palm, *dp_frames, *fingertip_frames))

        scalar.sort()
        layout = EmbodimentLayout(
            num_joint_q=len(builder.joint_q),
            num_joint_dof=builder.joint_dof_count,
            hands=tuple(hands),
            scalar_joints=ScalarJointLayout(
                q_ids=tuple(q_id for q_id, _, _ in scalar),
                dof_ids=tuple(dof_id for _, dof_id, _ in scalar),
                names=tuple(name for _, _, name in scalar),
            ),
            semantic_frames=tuple(semantic_frames),
        )
        if layout.scalar_joints.q_ids != tuple(range(7, layout.num_joint_q)):
            raise ValueError("G1 scalar q layout must follow its seven-coordinate floating base")
        if layout.scalar_joints.dof_ids != tuple(range(6, layout.num_joint_dof)):
            raise ValueError("G1 scalar DOF layout must follow its six-DOF floating base")
        if (
            layout.scalar_joints.names
            != G1_BODY_JOINT_NAMES[:22] + G1_DEX3_JOINT_NAMES[:7] + G1_BODY_JOINT_NAMES[22:] + G1_DEX3_JOINT_NAMES[7:]
        ):
            raise ValueError("G1 scalar joint order does not match the upstream model-12 Dex3 contract")
        return layout

    def bind_reference(
        self,
        builder: newton.ModelBuilder,
        layout: EmbodimentLayout,
        reference,
    ) -> RobotReferenceBinding:
        """Bind named G1 joints and the floating pelvis trajectory."""
        if not isinstance(reference, NamedRobotReference):
            raise TypeError("G1Dex3 requires a NamedRobotReference")
        joint_q, joint_target = build_named_joint_trajectory(
            builder,
            layout,
            tuple(reference.robot_joint_names()),
            np.asarray(reference.robot_joint_pos(), dtype=np.float32),
            joint_limit_tolerance=self.config.joint_limit_tolerance,
        )
        root_joint = _floating_root_joint(builder)
        q_start = builder.joint_q_start[root_joint]
        root_position = _finite_trajectory(reference.robot_root_pos_w(), "robot root position", 3)
        root_quaternion = _finite_trajectory(reference.robot_root_quat_w(), "robot root quaternion", 4)
        if root_position.shape[0] != joint_q.shape[0] or root_quaternion.shape[0] != joint_q.shape[0]:
            raise ValueError("robot root pose and joint trajectories must have the same frame count")
        norms = np.linalg.norm(root_quaternion, axis=1)
        if np.any(norms <= 1.0e-8):
            raise ValueError("robot root quaternions must have nonzero norm")
        root_quaternion = root_quaternion / norms[:, None]
        joint_q[:, q_start : q_start + 3] = root_position
        joint_q[:, q_start + 3 : q_start + 7] = root_quaternion[:, (1, 2, 3, 0)]
        binding = build_robot_reference_binding(builder, layout, joint_q, joint_target, reference.fps)
        if self.config.validate_palm_frames:
            _validate_palm_frames(binding, reference, self.config)
        return binding


def _arm_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_shoulder_pitch_joint",
        f"{side}_shoulder_roll_joint",
        f"{side}_shoulder_yaw_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_roll_joint",
        f"{side}_wrist_pitch_joint",
        f"{side}_wrist_yaw_joint",
    )


def _finger_joint_names(side: str) -> tuple[str, ...]:
    return tuple(
        f"{side}_hand_{digit}_{index}_joint"
        for digit, count in (("thumb", 3), ("middle", 2), ("index", 2))
        for index in range(count)
    )


def _hand_link_names(side: str) -> tuple[str, ...]:
    return (f"{side}_hand_palm_link", *(name.removesuffix("_joint") + "_link" for name in _finger_joint_names(side)))


def _semantic_body_names() -> tuple[str, ...]:
    return ("pelvis", *(name for side in _SIDES for name in _hand_link_names(side)))


def _capture_hand_geometry(
    builder: newton.ModelBuilder,
    body_ids: dict[str, int],
) -> dict[str, tuple[HandLinkGeometry, ...]]:
    shapes_by_body: dict[int, list[int]] = {}
    for shape_id, body_id in enumerate(builder.shape_body):
        shapes_by_body.setdefault(body_id, []).append(shape_id)
    geometry = {}
    for side in _SIDES:
        links = []
        for name in _hand_link_names(side):
            shape_ids = tuple(shapes_by_body.get(body_ids[name], ()))
            if not shape_ids:
                raise ValueError(f"G1 Dex3 hand link {name!r} has no collision or visual shapes")
            links.append(HandLinkGeometry(link_name=name, shape_ids=shape_ids, object_contact=True))
        geometry[side] = tuple(links)
    return geometry


def _replace_cylinders_with_capsules(builder: newton.ModelBuilder, shape_start: int) -> int:
    """Mirror Isaac's URDF cylinder conversion without changing dimensions or transforms."""
    replaced = 0
    for shape in range(shape_start, builder.shape_count):
        if int(builder.shape_type[shape]) == int(newton.GeoType.CYLINDER):
            builder.shape_type[shape] = int(newton.GeoType.CAPSULE)
            replaced += 1
    return replaced


def _fingertip_frame(side: str, digit: str, dp_frame: BodyFrame) -> BodyFrame:
    if dp_frame.body_to_frame_pos != (0.0, 0.0, 0.0) or dp_frame.body_to_frame_quat_xyzw != (0.0, 0.0, 0.0, 1.0):
        raise ValueError(f"G1 Dex3 terminal link {dp_frame.name!r} must remain a retained body")
    offset = (0.0, -0.035 if side == "left" else 0.035, 0.0) if digit == "thumb" else (0.035, 0.0, 0.0)
    return BodyFrame(name=f"{side}_{digit}_fingertip", body_id=dp_frame.body_id, body_to_frame_pos=offset)


def _floating_root_joint(builder: newton.ModelBuilder) -> int:
    joints = [joint for joint, joint_type in enumerate(builder.joint_type) if joint_type == newton.JointType.FREE]
    if len(joints) != 1:
        raise ValueError(f"G1 Dex3 requires exactly one floating base, got {len(joints)}")
    joint = joints[0]
    if builder.joint_dof_dim[joint] != (3, 3):
        raise ValueError("G1 floating base must have three linear and three angular DOFs")
    return joint


def _finite_trajectory(values, label: str, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite (frames, {width}) array, got {array.shape}")
    return np.ascontiguousarray(array)


def _validate_palm_frames(binding: RobotReferenceBinding, reference, config: G1Dex3Config) -> None:
    if not isinstance(reference, NamedFrameReference):
        raise TypeError("G1 Dex3 palm-frame validation requires a NamedFrameReference")
    names = tuple(reference.robot_frame_names())
    index = {name: position for position, name in enumerate(names)}
    required = tuple(f"{side}_hand_palm_link" for side in _SIDES)
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"G1 reference is missing palm frames: {missing}")
    target_position = np.asarray(reference.robot_frame_pos_w(), dtype=np.float64)
    target_quaternion = np.asarray(reference.robot_frame_quat_w(), dtype=np.float64)
    for side, name in zip(_SIDES, required, strict=True):
        hand = binding.hand(side)
        source_id = index[name]
        position_error = np.linalg.norm(
            np.asarray(hand.wrist_pos_w, dtype=np.float64) - target_position[:, source_id], axis=1
        )
        actual_quaternion = np.asarray(hand.wrist_quat_w, dtype=np.float64)
        expected_quaternion = target_quaternion[:, source_id]
        actual_quaternion /= np.linalg.norm(actual_quaternion, axis=1, keepdims=True)
        expected_quaternion /= np.linalg.norm(expected_quaternion, axis=1, keepdims=True)
        dot = np.abs(np.sum(actual_quaternion * expected_quaternion, axis=1))
        orientation_error = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
        if float(position_error.max()) > config.frame_position_tolerance:
            raise ValueError(f"{name} position differs from the reference by {float(position_error.max()):.9g} m")
        if float(orientation_error.max()) > config.frame_orientation_tolerance:
            raise ValueError(
                f"{name} orientation differs from the reference by {float(orientation_error.max()):.9g} rad"
            )


assert isinstance(G1Dex3(), Embodiment)
