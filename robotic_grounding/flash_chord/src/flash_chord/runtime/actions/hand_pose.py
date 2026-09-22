# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Residual hand-pose action processing and shared action protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
import warp as wp

from flash_chord.embodiments.base import EmbodimentLayout
from flash_chord.embodiments.binding import DeviceRobotReference, RobotReferenceBinding
from flash_chord.runtime.control import WristController
from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig
from flash_chord.runtime.residual import (
    InputMapping,
    InputMode,
    build_residual_mapping,
    filter_residual,
    map_residual_input,
)
from flash_chord.utils.quat import quat_mul_xyzw, quat_to_rotvec

if TYPE_CHECKING:
    from flash_chord.scene.builder import Scene


@runtime_checkable
class Action(Protocol):
    """Device action strategy consumed by :class:`BaseEnv`."""

    action_dim: int
    joint_target: wp.array

    def process(self, action: wp.array, timestep: wp.array, state=None) -> None:
        """Translate policy input and optional live state into persistent device control targets."""
        ...

    def prepare_control(self, control) -> None:
        """Write control values that are constant across one control frame."""
        ...

    def apply_control(self, state, control) -> None:
        """Write state-dependent control values for one physics substep."""
        ...

    def reset(self, reset_mask: wp.array) -> None:
        """Clear persistent state for selected worlds."""
        ...


@runtime_checkable
class ActionSpec(Protocol):
    """Setup-time action builder selected by configuration."""

    def build(
        self,
        scene: Scene,
        device=None,
    ) -> Action:
        """Bind runtime dependencies and return an action strategy."""
        ...


@runtime_checkable
class ActionDiagnostics(Protocol):
    """Exact policy-input semantics and named flattened action ranges."""

    action_dim: int
    input_mode: InputMode
    input_mapping: InputMapping
    block_names: tuple[str, ...]
    block_ranges: tuple[tuple[int, int], ...]


@runtime_checkable
class RegularizedAction(Action, Protocol):
    """Action strategy exposing standard L2 and rate-L2 terms."""

    action_l2: wp.array
    action_rate_l2: wp.array


@runtime_checkable
class PolicyAction(RegularizedAction, Protocol):
    """Action strategy exposing action-owned policy observation fields."""

    sides: tuple[str, ...]
    processed_dim: int
    raw_action: wp.array
    processed_target: wp.array


@runtime_checkable
class ReferenceOffsetAction(Action, Protocol):
    """Action whose reference conditioning accepts per-world joint-coordinate offsets."""

    def bind_reference_joint_q_offset(self, offset: wp.array) -> None:
        """Bind the environment-owned ``[W * q_per_world]`` offset buffer."""
        ...


@dataclass(frozen=True)
class ActionConfig:
    wrist_pos_scale: float = 0.05
    wrist_ori_scale: float = 0.15
    finger_joint_pos_scale: float = 0.15
    wrist_pos_clip: float = 0.2
    wrist_ori_clip: float = 1.0
    finger_joint_pos_clip: float = 1.0
    ema: float = 0.3
    wrist_kp: float = 25.0
    wrist_tau_max: float = 45.0
    delay: TargetDelayConfig = field(default_factory=TargetDelayConfig)
    input_mode: Literal["raw", "normalized"] = "raw"
    normalized_mapping: Literal["linear", "rational"] = "linear"

    def build(
        self,
        scene: Scene,
        device=None,
    ) -> Action:
        """Build the residual hand-pose action."""
        return ResidualHandPoseAction.build(
            scene,
            config=self,
            device=device,
        )


@dataclass(frozen=True)
class ActionLayout:
    """Policy and processed-target offsets plus simulation DOF mappings."""

    sides: tuple[str, ...]
    num_hands: int
    num_joint_dof: int
    action_dim: int
    processed_dim: int
    total_fingers: int
    hand_action_starts: tuple[int, ...]
    hand_processed_starts: tuple[int, ...]
    hand_finger_starts: tuple[int, ...]
    finger_counts: tuple[int, ...]
    wrist_pos_dof_ids: tuple[int, ...]
    wrist_ori_dof_ids: tuple[int, ...]
    finger_dof_ids: tuple[int, ...]

    @classmethod
    def from_embodiment(
        cls,
        layout: EmbodimentLayout,
        sides: tuple[str, ...] = ("right", "left"),
        num_joint_dof: int | None = None,
    ) -> "ActionLayout":
        if set(sides) != set(layout.sides) or len(sides) != len(layout.sides):
            raise ValueError(f"policy sides {sides} must match embodiment sides {layout.sides}")

        action_starts: list[int] = []
        processed_starts: list[int] = []
        finger_starts: list[int] = []
        finger_counts: list[int] = []
        wrist_pos_dof_ids: list[int] = []
        wrist_ori_dof_ids: list[int] = []
        finger_dof_ids: list[int] = []
        action_dim = 0
        processed_dim = 0
        total_fingers = 0
        for side in sides:
            hand = layout.hand(side)
            if len(hand.wrist_pos_dof_ids) != 3 or len(hand.wrist_orient_dof_ids) != 3:
                raise ValueError(f"{side} policy action requires a 3-DOF position and 3-DOF orientation wrist")
            num_fingers = len(hand.finger_dof_ids)
            action_starts.append(action_dim)
            processed_starts.append(processed_dim)
            finger_starts.append(total_fingers)
            finger_counts.append(num_fingers)
            wrist_pos_dof_ids.extend(hand.wrist_pos_dof_ids)
            wrist_ori_dof_ids.extend(hand.wrist_orient_dof_ids)
            finger_dof_ids.extend(hand.finger_dof_ids)
            action_dim += 6 + num_fingers
            processed_dim += 7 + num_fingers
            total_fingers += num_fingers

        env_joint_dof = layout.num_joint_dof if num_joint_dof is None else num_joint_dof
        if env_joint_dof < layout.num_joint_dof:
            raise ValueError(f"environment has {env_joint_dof} DOFs; embodiment requires {layout.num_joint_dof}")
        return cls(
            sides=sides,
            num_hands=len(sides),
            num_joint_dof=env_joint_dof,
            action_dim=action_dim,
            processed_dim=processed_dim,
            total_fingers=total_fingers,
            hand_action_starts=tuple(action_starts),
            hand_processed_starts=tuple(processed_starts),
            hand_finger_starts=tuple(finger_starts),
            finger_counts=tuple(finger_counts),
            wrist_pos_dof_ids=tuple(wrist_pos_dof_ids),
            wrist_ori_dof_ids=tuple(wrist_ori_dof_ids),
            finger_dof_ids=tuple(finger_dof_ids),
        )


@wp.kernel
def process_residual_action(
    action: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    num_frames: int,
    ref_wrist_pos_w: wp.array(dtype=wp.vec3),
    ref_wrist_quat_w: wp.array(dtype=wp.quat),
    ref_finger_joint_pos: wp.array(dtype=wp.float32),
    num_hands: int,
    action_dim: int,
    processed_dim: int,
    total_fingers: int,
    num_joint_dof: int,
    hand_action_starts: wp.array(dtype=wp.int32),
    hand_processed_starts: wp.array(dtype=wp.int32),
    hand_finger_starts: wp.array(dtype=wp.int32),
    finger_counts: wp.array(dtype=wp.int32),
    wrist_pos_dof_ids: wp.array(dtype=wp.int32),
    wrist_ori_dof_ids: wp.array(dtype=wp.int32),
    finger_dof_ids: wp.array(dtype=wp.int32),
    input_mapping: int,
    input_scale: wp.array(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    clip: wp.array(dtype=wp.float32),
    ema: float,
    raw_action: wp.array(dtype=wp.float32),
    filtered_action: wp.array(dtype=wp.float32),
    processed_target: wp.array(dtype=wp.float32),
    joint_target: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    action_rate_l2: wp.array(dtype=wp.float32),
) -> None:
    """Process one world's action and gather its reference targets."""
    world = wp.tid()
    action_base = world * action_dim
    l2 = float(0.0)
    rate_l2 = float(0.0)
    for i in range(action_dim):
        index = action_base + i
        value = map_residual_input(action[index], input_mapping, input_scale[i])
        previous_raw = raw_action[index]
        l2 += value * value
        delta = value - previous_raw
        rate_l2 += delta * delta
        raw_action[index] = value
        filtered_action[index] = filter_residual(
            value,
            filtered_action[index],
            scale[i],
            clip[i],
            ema,
        )
    action_l2[world] = l2
    action_rate_l2[world] = rate_l2

    frame = timestep[world]
    if frame < 0:
        frame = 0
    if frame >= num_frames:
        frame = num_frames - 1

    for hand in range(num_hands):
        hand_action = hand_action_starts[hand]
        filtered_base = action_base + hand_action
        processed_base = world * processed_dim + hand_processed_starts[hand]
        reference_hand = frame * num_hands + hand

        target_pos_w = ref_wrist_pos_w[reference_hand] + wp.vec3(
            filtered_action[filtered_base],
            filtered_action[filtered_base + 1],
            filtered_action[filtered_base + 2],
        )
        residual_quat = wp.quat_rpy(
            filtered_action[filtered_base + 3],
            filtered_action[filtered_base + 4],
            filtered_action[filtered_base + 5],
        )
        target_quat_w = wp.normalize(quat_mul_xyzw(ref_wrist_quat_w[reference_hand], residual_quat))
        target_rotvec_w = quat_to_rotvec(target_quat_w)
        processed_target[processed_base] = target_pos_w[0]
        processed_target[processed_base + 1] = target_pos_w[1]
        processed_target[processed_base + 2] = target_pos_w[2]
        processed_target[processed_base + 3] = target_quat_w[3]
        processed_target[processed_base + 4] = target_quat_w[0]
        processed_target[processed_base + 5] = target_quat_w[1]
        processed_target[processed_base + 6] = target_quat_w[2]

        joint_base = world * num_joint_dof
        for axis in range(3):
            joint_target[joint_base + wrist_pos_dof_ids[hand * 3 + axis]] = target_pos_w[axis]
            joint_target[joint_base + wrist_ori_dof_ids[hand * 3 + axis]] = target_rotvec_w[axis]

        finger_start = hand_finger_starts[hand]
        num_fingers = finger_counts[hand]
        reference_finger_base = frame * total_fingers + finger_start
        for finger in range(num_fingers):
            reference_target = ref_finger_joint_pos[reference_finger_base + finger]
            target = reference_target + filtered_action[filtered_base + 6 + finger]
            processed_target[processed_base + 7 + finger] = target
            joint_target[joint_base + finger_dof_ids[finger_start + finger]] = target


@wp.kernel
def reset_action_state(
    reset_mask: wp.array(dtype=wp.int32),
    num_hands: int,
    action_dim: int,
    processed_dim: int,
    hand_processed_starts: wp.array(dtype=wp.int32),
    raw_action: wp.array(dtype=wp.float32),
    filtered_action: wp.array(dtype=wp.float32),
    processed_target: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    action_rate_l2: wp.array(dtype=wp.float32),
) -> None:
    """Clear action state for selected worlds."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    action_base = world * action_dim
    for i in range(action_dim):
        raw_action[action_base + i] = 0.0
        filtered_action[action_base + i] = 0.0
    processed_base = world * processed_dim
    for i in range(processed_dim):
        processed_target[processed_base + i] = 0.0
    for hand in range(num_hands):
        processed_target[processed_base + hand_processed_starts[hand] + 3] = 1.0
    action_l2[world] = 0.0
    action_rate_l2[world] = 0.0


@dataclass
class ActionBuffers:
    """Device mappings and persistent state for residual hand-pose actions."""

    layout: ActionLayout
    reference: DeviceRobotReference
    world_count: int
    hand_action_starts: wp.array
    hand_processed_starts: wp.array
    hand_finger_starts: wp.array
    finger_counts: wp.array
    wrist_pos_dof_ids: wp.array
    wrist_ori_dof_ids: wp.array
    finger_dof_ids: wp.array
    input_mode: Literal["raw", "normalized"]
    normalized_mapping: Literal["linear", "rational"]
    input_mapping: Literal["identity", "linear", "rational"]
    input_mapping_code: int
    input_scale_values: tuple[float, ...]
    input_scale: wp.array
    scale: wp.array
    clip: wp.array
    raw_action: wp.array
    filtered_action: wp.array
    processed_target: wp.array
    joint_target: wp.array
    action_l2: wp.array
    action_rate_l2: wp.array
    ema: float

    @classmethod
    def build(
        cls,
        embodiment_layout: EmbodimentLayout,
        reference: RobotReferenceBinding,
        world_count: int,
        config: ActionConfig | None = None,
        sides: tuple[str, ...] = ("right", "left"),
        num_joint_dof: int | None = None,
        device=None,
    ) -> "ActionBuffers":
        config = config or ActionConfig()
        for name in (
            "wrist_pos_scale",
            "wrist_ori_scale",
            "finger_joint_pos_scale",
            "wrist_pos_clip",
            "wrist_ori_clip",
            "finger_joint_pos_clip",
        ):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        layout = ActionLayout.from_embodiment(embodiment_layout, sides=sides, num_joint_dof=num_joint_dof)
        action_reference = DeviceRobotReference.build(
            reference,
            embodiment_layout,
            sides=layout.sides,
            device=device,
        )

        scale = np.empty(layout.action_dim, dtype=np.float32)
        clip = np.empty(layout.action_dim, dtype=np.float32)
        for hand, start in enumerate(layout.hand_action_starts):
            num_fingers = layout.finger_counts[hand]
            scale[start : start + 3] = config.wrist_pos_scale
            scale[start + 3 : start + 6] = config.wrist_ori_scale
            scale[start + 6 : start + 6 + num_fingers] = config.finger_joint_pos_scale
            clip[start : start + 3] = config.wrist_pos_clip
            clip[start + 3 : start + 6] = config.wrist_ori_clip
            clip[start + 6 : start + 6 + num_fingers] = config.finger_joint_pos_clip
        mapping = build_residual_mapping(
            scale,
            clip,
            ema=config.ema,
            input_mode=config.input_mode,
            normalized_mapping=config.normalized_mapping,
        )

        i32 = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
        return cls(
            layout=layout,
            reference=action_reference,
            world_count=world_count,
            hand_action_starts=i32(layout.hand_action_starts),
            hand_processed_starts=i32(layout.hand_processed_starts),
            hand_finger_starts=i32(layout.hand_finger_starts),
            finger_counts=i32(layout.finger_counts),
            wrist_pos_dof_ids=i32(layout.wrist_pos_dof_ids),
            wrist_ori_dof_ids=i32(layout.wrist_ori_dof_ids),
            finger_dof_ids=i32(layout.finger_dof_ids),
            input_mode=mapping.input_mode,
            normalized_mapping=mapping.normalized_mapping,
            input_mapping=mapping.input_mapping,
            input_mapping_code=mapping.input_mapping_code,
            input_scale_values=tuple(float(value) for value in mapping.input_scale),
            input_scale=wp.array(mapping.input_scale, dtype=wp.float32, device=device),
            scale=wp.array(mapping.scale, dtype=wp.float32, device=device),
            clip=wp.array(mapping.clip, dtype=wp.float32, device=device),
            raw_action=wp.zeros(world_count * layout.action_dim, dtype=wp.float32, device=device),
            filtered_action=wp.zeros(world_count * layout.action_dim, dtype=wp.float32, device=device),
            processed_target=wp.zeros(world_count * layout.processed_dim, dtype=wp.float32, device=device),
            joint_target=wp.zeros(world_count * layout.num_joint_dof, dtype=wp.float32, device=device),
            action_l2=wp.zeros(world_count, dtype=wp.float32, device=device),
            action_rate_l2=wp.zeros(world_count, dtype=wp.float32, device=device),
            ema=mapping.ema,
        )

    def process(self, action: wp.array, timestep: wp.array, state=None) -> None:
        """Process ``action`` against each world's current reference timestep."""
        layout = self.layout
        wp.launch(
            process_residual_action,
            dim=self.world_count,
            inputs=[
                action,
                timestep,
                self.reference.num_frames,
                self.reference.wrist_pos_w,
                self.reference.wrist_quat_w,
                self.reference.finger_joint_pos,
                layout.num_hands,
                layout.action_dim,
                layout.processed_dim,
                layout.total_fingers,
                layout.num_joint_dof,
                self.hand_action_starts,
                self.hand_processed_starts,
                self.hand_finger_starts,
                self.finger_counts,
                self.wrist_pos_dof_ids,
                self.wrist_ori_dof_ids,
                self.finger_dof_ids,
                self.input_mapping_code,
                self.input_scale,
                self.scale,
                self.clip,
                self.ema,
            ],
            outputs=[
                self.raw_action,
                self.filtered_action,
                self.processed_target,
                self.joint_target,
                self.action_l2,
                self.action_rate_l2,
            ],
        )

    def reset(self, reset_mask: wp.array) -> None:
        """Clear persistent action state where ``reset_mask`` is nonzero."""
        wp.launch(
            reset_action_state,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.layout.num_hands,
                self.layout.action_dim,
                self.layout.processed_dim,
                self.hand_processed_starts,
            ],
            outputs=[
                self.raw_action,
                self.filtered_action,
                self.processed_target,
                self.action_l2,
                self.action_rate_l2,
            ],
        )


@dataclass
class ResidualHandPoseAction:
    """Reference-relative wrist and finger action with Sharpa wrist control."""

    buffers: ActionBuffers
    wrist: WristController
    delay: TargetDelayBuffer | None = None

    @classmethod
    def build(
        cls,
        scene: Scene,
        config: ActionConfig | None = None,
        device=None,
    ) -> "ResidualHandPoseAction":
        """Build the residual action strategy for ``scene``."""
        config = config or ActionConfig()
        for name in ("wrist_kp", "wrist_tau_max"):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        num_joint_dof = scene.model.joint_dof_count // scene.world_count
        delay = None
        if config.delay.max_steps > 0:
            delay = TargetDelayBuffer.build(
                world_count=scene.world_count,
                num_joint_dof=num_joint_dof,
                delayed_dof_ids=tuple(range(scene.layout.num_joint_dof)),
                config=config.delay,
                device=device,
            )
        return cls(
            buffers=ActionBuffers.build(
                scene.layout,
                scene.robot_reference,
                world_count=scene.world_count,
                config=config,
                num_joint_dof=num_joint_dof,
                device=device,
            ),
            wrist=WristController.from_layout(
                scene.layout,
                scene.model,
                scene.world_count,
                kp=config.wrist_kp,
                tau_max=config.wrist_tau_max,
                device=device,
            ),
            delay=delay,
        )

    @property
    def action_dim(self) -> int:
        return self.buffers.layout.action_dim

    @property
    def processed_dim(self) -> int:
        return self.buffers.layout.processed_dim

    @property
    def sides(self) -> tuple[str, ...]:
        return self.buffers.layout.sides

    @property
    def input_mode(self) -> Literal["raw", "normalized"]:
        """Configured policy-input interpretation."""
        return self.buffers.input_mode

    @property
    def normalized_mapping(self) -> Literal["linear", "rational"]:
        """Configured mapping for normalized policy input."""
        return self.buffers.normalized_mapping

    @property
    def input_mapping(self) -> Literal["identity", "linear", "rational"]:
        """Effective policy-to-residual mapping, normalized for raw input."""
        return self.buffers.input_mapping

    @property
    def input_scale_values(self) -> tuple[float, ...]:
        """Per-dimension normalized endpoint bounds in raw residual units."""
        return self.buffers.input_scale_values

    @property
    def block_names(self) -> tuple[str, ...]:
        """Semantic action blocks in flattened policy order."""
        return tuple(
            name
            for side in self.sides
            for name in (
                f"{side}_wrist_position_residual_normalized",
                f"{side}_wrist_orientation_residual_normalized",
                f"{side}_finger_joint_residual_normalized",
            )
        )

    @property
    def block_ranges(self) -> tuple[tuple[int, int], ...]:
        """Half-open flattened ranges corresponding to :attr:`block_names`."""
        ranges = []
        for start, finger_count in zip(
            self.buffers.layout.hand_action_starts,
            self.buffers.layout.finger_counts,
            strict=True,
        ):
            ranges.extend(((start, start + 3), (start + 3, start + 6), (start + 6, start + 6 + finger_count)))
        return tuple(ranges)

    @property
    def joint_target(self) -> wp.array:
        return self.buffers.joint_target

    @property
    def raw_action(self) -> wp.array:
        return self.buffers.raw_action

    @property
    def filtered_action(self) -> wp.array:
        return self.buffers.filtered_action

    @property
    def processed_target(self) -> wp.array:
        return self.buffers.processed_target

    @property
    def action_l2(self) -> wp.array:
        return self.buffers.action_l2

    @property
    def action_rate_l2(self) -> wp.array:
        return self.buffers.action_rate_l2

    def process(self, action: wp.array, timestep: wp.array, state=None) -> None:
        self.buffers.process(action, timestep, state)

    def prepare_control(self, control) -> None:
        if self.delay is None:
            self.wrist.write_position_targets(control, self.joint_target)

    def apply_control(self, state, control) -> None:
        target = self.joint_target
        if self.delay is not None:
            target = self.delay.advance(target)
            self.wrist.write_position_targets(control, target)
        self.wrist.apply_wrist_effort(state, control, target)

    def reset(self, reset_mask: wp.array) -> None:
        self.buffers.reset(reset_mask)
        if self.delay is not None:
            self.delay.reset(reset_mask)
