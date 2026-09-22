# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup-time robot reference data in exact simulation order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import newton
import numpy as np
import warp as wp

from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout

HandKeypointFrame = Literal["dp", "fingertip"]


@dataclass(frozen=True)
class BoundHandReference:
    """One hand's FK-derived reference trajectories in layout order."""

    side: str
    wrist_pos_w: np.ndarray
    wrist_quat_w: np.ndarray
    arm_joint_pos: np.ndarray
    finger_joint_pos: np.ndarray
    dp_pos_w: np.ndarray
    dp_quat_w: np.ndarray
    fingertip_pos_w: np.ndarray
    fingertip_quat_w: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.side, str) or not self.side:
            raise ValueError(f"bound hand side must be a nonempty string, got {self.side!r}")
        wrist_pos = _finite_array(self.wrist_pos_w, f"{self.side} wrist_pos_w", 2, (3,))
        num_frames = wrist_pos.shape[0]
        values = {
            "wrist_pos_w": wrist_pos,
            "wrist_quat_w": _normalized_quaternion_array(
                self.wrist_quat_w,
                f"{self.side} wrist_quat_w",
                2,
                (4,),
            ),
            "arm_joint_pos": _finite_array(
                self.arm_joint_pos,
                f"{self.side} arm_joint_pos",
                2,
            ),
            "finger_joint_pos": _finite_array(
                self.finger_joint_pos,
                f"{self.side} finger_joint_pos",
                2,
            ),
            "dp_pos_w": _finite_array(self.dp_pos_w, f"{self.side} dp_pos_w", 3, (3,)),
            "dp_quat_w": _normalized_quaternion_array(
                self.dp_quat_w,
                f"{self.side} dp_quat_w",
                3,
                (4,),
            ),
            "fingertip_pos_w": _finite_array(
                self.fingertip_pos_w,
                f"{self.side} fingertip_pos_w",
                3,
                (3,),
            ),
            "fingertip_quat_w": _normalized_quaternion_array(
                self.fingertip_quat_w,
                f"{self.side} fingertip_quat_w",
                3,
                (4,),
            ),
        }
        for name, array in values.items():
            if array.shape[0] != num_frames:
                raise ValueError(f"bound {self.side} {name} has {array.shape[0]} frames; expected {num_frames}")
            object.__setattr__(self, name, _readonly(array))
        if values["dp_pos_w"].shape[:2] != values["dp_quat_w"].shape[:2]:
            raise ValueError(f"bound {self.side} DP position/quaternion counts must match")
        if values["fingertip_pos_w"].shape[:2] != values["fingertip_quat_w"].shape[:2]:
            raise ValueError(f"bound {self.side} fingertip position/quaternion counts must match")
        if values["dp_pos_w"].shape[1] != values["fingertip_pos_w"].shape[1]:
            raise ValueError(f"bound {self.side} DP and fingertip counts must match")

    def keypoint_pos_w(self, frame: HandKeypointFrame) -> np.ndarray:
        """Return palm plus five digit points for the selected semantic geometry."""
        if frame == "dp":
            digits = self.dp_pos_w
        elif frame == "fingertip":
            digits = self.fingertip_pos_w
        else:
            raise ValueError(f"hand keypoint frame must be 'dp' or 'fingertip', got {frame!r}")
        return np.concatenate((self.wrist_pos_w[:, None, :], digits), axis=1)


@dataclass(frozen=True)
class RobotReferenceBinding:
    """A raw robot reference bound once to one exact simulation layout."""

    num_frames: int
    fps: float
    num_joint_q: int
    num_joint_dof: int
    joint_q: np.ndarray
    joint_target: np.ndarray
    hands: tuple[BoundHandReference, ...]

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"robot reference must contain at least one frame, got {self.num_frames}")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"robot reference fps must be positive and finite, got {self.fps}")
        if self.num_joint_q < 0 or self.num_joint_dof < 0:
            raise ValueError("robot reference joint dimensions must be nonnegative")
        object.__setattr__(self, "num_frames", int(self.num_frames))
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "num_joint_q", int(self.num_joint_q))
        object.__setattr__(self, "num_joint_dof", int(self.num_joint_dof))
        joint_q = _trajectory(self.joint_q, "joint_q", self.num_joint_q)
        joint_target = _trajectory(self.joint_target, "joint_target", self.num_joint_dof)
        if joint_q.shape[0] != self.num_frames or joint_target.shape[0] != self.num_frames:
            raise ValueError(
                f"robot reference declares {self.num_frames} frames but joint_q/joint_target have "
                f"{joint_q.shape[0]}/{joint_target.shape[0]}"
            )
        object.__setattr__(self, "joint_q", _readonly(joint_q))
        object.__setattr__(self, "joint_target", _readonly(joint_target))
        object.__setattr__(self, "hands", tuple(self.hands))
        sides = tuple(hand.side for hand in self.hands)
        if not sides or len(set(sides)) != len(sides):
            raise ValueError(f"bound robot hand sides must be unique and nonempty, got {sides}")
        for hand in self.hands:
            if hand.wrist_pos_w.shape[0] != self.num_frames:
                raise ValueError(
                    f"bound {hand.side} hand has {hand.wrist_pos_w.shape[0]} frames; expected {self.num_frames}"
                )

    def validate(self, layout: EmbodimentLayout) -> None:
        """Validate trajectory widths and semantic counts against one exact layout."""
        if self.num_joint_q != layout.num_joint_q or self.num_joint_dof != layout.num_joint_dof:
            raise ValueError(
                "bound robot dimensions "
                f"({self.num_joint_q}, {self.num_joint_dof}) do not match layout "
                f"({layout.num_joint_q}, {layout.num_joint_dof})"
            )
        sides = tuple(hand.side for hand in self.hands)
        if set(sides) != set(layout.sides) or len(sides) != len(layout.sides):
            raise ValueError(f"bound robot sides {sides} must match embodiment sides {layout.sides}")
        for side in layout.sides:
            hand = self.hand(side)
            hand_layout = layout.hand(side)
            expected = {
                "arm_joint_pos": len(hand_layout.arm_q_ids),
                "finger_joint_pos": len(hand_layout.finger_q_ids),
                "dp_pos_w": len(hand_layout.dp_frames),
                "fingertip_pos_w": len(hand_layout.fingertip_frames),
            }
            for name, width in expected.items():
                actual = getattr(hand, name).shape[1]
                if actual != width:
                    raise ValueError(f"bound {side} {name} has width {actual}; expected {width}")

    def hand(self, side: str) -> BoundHandReference:
        for hand in self.hands:
            if hand.side == side:
                return hand
        raise KeyError(f"no bound hand for side {side!r}; have {[hand.side for hand in self.hands]}")


@dataclass(frozen=True)
class DeviceRobotReference:
    """Device view of a bound trajectory in one explicit hand order."""

    num_frames: int
    num_joint_dof: int
    sides: tuple[str, ...]
    arm_counts: tuple[int, ...]
    finger_counts: tuple[int, ...]
    joint_q: wp.array
    joint_target: wp.array
    wrist_pos_w: wp.array
    wrist_quat_w: wp.array
    arm_joint_pos: wp.array
    finger_joint_pos: wp.array

    @classmethod
    def build(
        cls,
        reference: RobotReferenceBinding,
        layout: EmbodimentLayout,
        sides: tuple[str, ...] | None = None,
        device=None,
    ) -> "DeviceRobotReference":
        sides = layout.sides if sides is None else tuple(sides)
        if set(sides) != set(layout.sides) or len(sides) != len(layout.sides):
            raise ValueError(f"device reference sides {sides} must match embodiment sides {layout.sides}")
        reference.validate(layout)

        hands = tuple(reference.hand(side) for side in sides)
        arm_joints = layout.select_scalar_joints(sides, ("arm",), require_names=True)
        finger_joints = layout.select_scalar_joints(sides, ("finger",))
        arm_counts = arm_joints.side_counts
        finger_counts = finger_joints.side_counts
        for hand, arm_count, finger_count in zip(hands, arm_counts, finger_counts, strict=True):
            expected = {
                "wrist_pos_w": (reference.num_frames, 3),
                "wrist_quat_w": (reference.num_frames, 4),
                "arm_joint_pos": (reference.num_frames, arm_count),
                "finger_joint_pos": (reference.num_frames, finger_count),
            }
            for name, shape in expected.items():
                actual = getattr(hand, name).shape
                if actual != shape:
                    raise ValueError(f"bound {hand.side} {name} has shape {actual}; expected {shape}")

        wrist_pos_w = np.stack([hand.wrist_pos_w for hand in hands], axis=1)
        wrist_quat_wxyz = np.stack([hand.wrist_quat_w for hand in hands], axis=1)
        arm_joint_pos = np.concatenate([hand.arm_joint_pos for hand in hands], axis=1)
        finger_joint_pos = np.concatenate([hand.finger_joint_pos for hand in hands], axis=1)
        return cls(
            num_frames=reference.num_frames,
            num_joint_dof=reference.num_joint_dof,
            sides=sides,
            arm_counts=arm_counts,
            finger_counts=finger_counts,
            joint_q=wp.array(reference.joint_q, dtype=wp.float32, device=device),
            joint_target=wp.array(reference.joint_target.reshape(-1), dtype=wp.float32, device=device),
            wrist_pos_w=wp.array(wrist_pos_w.reshape(-1, 3), dtype=wp.vec3, device=device),
            wrist_quat_w=wp.array(
                wrist_quat_wxyz[..., (1, 2, 3, 0)].reshape(-1, 4),
                dtype=wp.quat,
                device=device,
            ),
            arm_joint_pos=wp.array(arm_joint_pos.reshape(-1), dtype=wp.float32, device=device),
            finger_joint_pos=wp.array(finger_joint_pos.reshape(-1), dtype=wp.float32, device=device),
        )


def build_named_joint_trajectory(
    builder: newton.ModelBuilder,
    layout: EmbodimentLayout,
    source_names: tuple[str, ...] | list[str],
    source_joint_pos: np.ndarray,
    *,
    joint_limit_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a named scalar-joint subset while preserving unmapped coordinates."""
    if not np.isfinite(joint_limit_tolerance) or joint_limit_tolerance < 0.0:
        raise ValueError(f"joint limit tolerance must be nonnegative and finite, got {joint_limit_tolerance}")
    source_names = tuple(source_names)
    invalid_source_names = [name for name in source_names if not isinstance(name, str) or not name]
    if invalid_source_names:
        raise ValueError(f"source joint names must be nonempty strings, got {invalid_source_names}")
    duplicate_source = _duplicates(source_names)
    if duplicate_source:
        raise ValueError(f"named robot reference has duplicate joints: {duplicate_source}")

    if layout.scalar_joints is None:
        raise ValueError("embodiment layout does not define named scalar joints")
    scalar_joints = layout.scalar_joints
    expected = tuple(zip(scalar_joints.q_ids, scalar_joints.dof_ids, scalar_joints.names, strict=True))
    expected_names = scalar_joints.names
    missing = sorted(set(expected_names) - set(source_names))
    unexpected = sorted(set(source_names) - set(expected_names))
    if missing or unexpected:
        raise ValueError(f"named robot joint mismatch: missing={missing}, unexpected={unexpected}")
    source_joint_pos = _trajectory(source_joint_pos, "source_joint_pos", len(source_names))
    if len(builder.joint_q) != layout.num_joint_q or len(builder.joint_target_pos) != layout.num_joint_dof:
        raise ValueError("binding builder defaults do not match the embodiment layout")

    source_index = {name: index for index, name in enumerate(source_names)}
    joint_q = np.tile(np.asarray(builder.joint_q, dtype=np.float32), (source_joint_pos.shape[0], 1))
    joint_target = np.tile(
        np.asarray(builder.joint_target_pos, dtype=np.float32),
        (source_joint_pos.shape[0], 1),
    )
    for q_id, dof_id, name in expected:
        values = source_joint_pos[:, source_index[name]]
        joint_q[:, q_id] = values
        joint_target[:, dof_id] = values

    joint_lower = np.asarray(builder.joint_limit_lower, dtype=np.float32)
    joint_upper = np.asarray(builder.joint_limit_upper, dtype=np.float32)
    if joint_lower.shape != (layout.num_joint_dof,) or joint_upper.shape != (layout.num_joint_dof,):
        raise ValueError(
            "builder joint limits do not match the embodiment DOF layout: "
            f"lower={joint_lower.shape}, upper={joint_upper.shape}, expected=({layout.num_joint_dof},)"
        )
    if np.any(np.isnan(joint_lower)) or np.any(np.isnan(joint_upper)) or np.any(joint_lower > joint_upper):
        raise ValueError("builder joint limits must be ordered and contain no NaN values")

    scalar_dof_ids = np.asarray(scalar_joints.dof_ids, dtype=np.int64)
    if scalar_dof_ids.size == 0:
        return joint_q, joint_target
    lower = joint_lower[scalar_dof_ids]
    upper = joint_upper[scalar_dof_ids]
    values = joint_target[:, scalar_dof_ids]
    lower_excess = lower[None, :] - values
    upper_excess = values - upper[None, :]
    excess = np.maximum(np.maximum(lower_excess, upper_excess), 0.0)
    max_excess = float(np.max(excess))
    if max_excess > joint_limit_tolerance:
        frame, joint = np.unravel_index(np.argmax(excess), excess.shape)
        value = float(values[frame, joint])
        below = lower_excess[frame, joint] >= upper_excess[frame, joint]
        bound = float(lower[joint] if below else upper[joint])
        bound_name = "lower" if below else "upper"
        raise ValueError(
            f"named joint {scalar_joints.names[joint]!r} at frame {frame} has value {value:.9g} "
            f"outside its {bound_name} limit {bound:.9g} by {max_excess:.9g}; "
            f"tolerance is {joint_limit_tolerance:.9g}"
        )
    joint_target[:, scalar_dof_ids] = np.clip(values, lower, upper)
    return joint_q, joint_target


def build_robot_reference_binding(
    builder: newton.ModelBuilder,
    layout: EmbodimentLayout,
    joint_q: np.ndarray,
    joint_target: np.ndarray,
    fps: float,
    *,
    device: str = "cpu",
) -> RobotReferenceBinding:
    """Derive semantic hand targets from one simulation-ordered robot trajectory."""
    joint_q = _trajectory(joint_q, "joint_q", layout.num_joint_q)
    joint_target = _trajectory(joint_target, "joint_target", layout.num_joint_dof)
    if joint_target.shape[0] != joint_q.shape[0]:
        raise ValueError(f"joint_q has {joint_q.shape[0]} frames but joint_target has {joint_target.shape[0]}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"reference fps must be positive and finite, got {fps}")
    if builder.joint_coord_count != layout.num_joint_q or builder.joint_dof_count != layout.num_joint_dof:
        raise ValueError("binding builder must contain exactly the embodiment's robot joints")

    body_q = _reference_body_q(builder, joint_q, device=device)
    hands = tuple(_bind_hand(body_q, layout.hand(side), joint_q) for side in layout.sides)
    return RobotReferenceBinding(
        num_frames=joint_q.shape[0],
        fps=float(fps),
        num_joint_q=layout.num_joint_q,
        num_joint_dof=layout.num_joint_dof,
        joint_q=_readonly(joint_q),
        joint_target=_readonly(joint_target),
        hands=hands,
    )


def _bind_hand(body_q: np.ndarray, hand, joint_q: np.ndarray) -> BoundHandReference:
    wrist_pos, wrist_quat = _frame_poses(body_q, (hand.palm_frame,))
    dp_pos, dp_quat = _frame_poses(body_q, hand.dp_frames)
    fingertip_pos, fingertip_quat = _frame_poses(body_q, hand.fingertip_frames)
    return BoundHandReference(
        side=hand.side,
        wrist_pos_w=_readonly(wrist_pos[:, 0]),
        wrist_quat_w=_readonly(_xyzw_to_wxyz(wrist_quat[:, 0])),
        arm_joint_pos=_readonly(joint_q[:, hand.arm_q_ids]),
        finger_joint_pos=_readonly(joint_q[:, hand.finger_q_ids]),
        dp_pos_w=_readonly(dp_pos),
        dp_quat_w=_readonly(_xyzw_to_wxyz(dp_quat)),
        fingertip_pos_w=_readonly(fingertip_pos),
        fingertip_quat_w=_readonly(_xyzw_to_wxyz(fingertip_quat)),
    )


def _reference_body_q(builder: newton.ModelBuilder, joint_q: np.ndarray, device: str) -> np.ndarray:
    """Evaluate every reference row through one reusable one-world Newton model."""
    model = builder.finalize(device=device, skip_all_validations=True)
    state = model.state()
    state.joint_qd.zero_()
    bodies_per_frame = builder.body_count
    body_q = np.empty((joint_q.shape[0], bodies_per_frame, 7), dtype=np.float32)
    for frame, frame_joint_q in enumerate(joint_q):
        state.joint_q.assign(frame_joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        body_q[frame] = np.asarray(state.body_q.numpy(), dtype=np.float32).reshape(bodies_per_frame, 7)
    return body_q


def _frame_poses(body_q: np.ndarray, frames: tuple[BodyFrame, ...]) -> tuple[np.ndarray, np.ndarray]:
    position = np.empty((body_q.shape[0], len(frames), 3), dtype=np.float32)
    quaternion = np.empty((body_q.shape[0], len(frames), 4), dtype=np.float32)
    for index, frame in enumerate(frames):
        body = body_q[:, frame.body_id]
        body_pos = body[:, :3]
        body_quat = body[:, 3:]
        local_pos = np.asarray(frame.body_to_frame_pos, dtype=np.float32)
        local_quat = np.asarray(frame.body_to_frame_quat_xyzw, dtype=np.float32)
        position[:, index] = body_pos + _quat_rotate_xyzw(body_quat, local_pos)
        quaternion[:, index] = _quat_mul_xyzw(body_quat, local_quat)
    return position, quaternion


def _quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, :3]
    w = quaternion[:, 3:4]
    first = np.cross(xyz, vector)
    return vector + 2.0 * (w * first + np.cross(xyz, first))


def _quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_xyz = left[:, :3]
    right_xyz = right[:3]
    xyz = left[:, 3:4] * right_xyz + right[3] * left_xyz + np.cross(left_xyz, right_xyz)
    w = left[:, 3] * right[3] - np.sum(left_xyz * right_xyz, axis=1)
    return np.concatenate((xyz, w[:, None]), axis=1)


def _xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return quaternion[..., (3, 0, 1, 2)]


def _finite_array(
    values: np.ndarray,
    name: str,
    ndim: int,
    trailing_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != ndim or (trailing_shape is not None and values.shape[-len(trailing_shape) :] != trailing_shape):
        suffix = "" if trailing_shape is None else f" ending in {trailing_shape}"
        raise ValueError(f"{name} must be a rank-{ndim} array{suffix}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(values)


def _normalized_quaternion_array(
    values: np.ndarray,
    name: str,
    ndim: int,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    values = _finite_array(values, name, ndim, trailing_shape)
    norms = np.linalg.norm(values, axis=-1)
    if np.any(norms <= 1.0e-8):
        raise ValueError(f"{name} quaternions must have nonzero norm")
    if values.size:
        values = values / norms[..., None]
    return np.ascontiguousarray(values, dtype=np.float32)


def _trajectory(values: np.ndarray, name: str, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (frames, {width}), got {values.shape}")
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain at least one finite frame")
    return np.ascontiguousarray(values)


def _readonly(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    values.setflags(write=False)
    return values


def _duplicates(names: tuple[str, ...]) -> list[str]:
    return sorted(name for name in set(names) if names.count(name) > 1)
