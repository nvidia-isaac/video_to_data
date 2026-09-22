# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-relative scalar-joint position action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import newton
import numpy as np
import warp as wp

from flash_chord.embodiments.base import ScalarJointSelection
from flash_chord.embodiments.binding import DeviceRobotReference
from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig
from flash_chord.runtime.residual import build_residual_mapping, filter_residual, map_residual_input

if TYPE_CHECKING:
    from flash_chord.scene.builder import Scene


@dataclass(frozen=True)
class FinalTargetTransformConfig:
    """Optional causal transform applied to the complete joint target."""

    mode: Literal["none", "ema", "smooth_slew"] = "none"
    arm_ema: float = 0.0
    finger_ema: float = 0.0
    arm_slew: float = 1.0
    finger_slew: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("none", "ema", "smooth_slew"):
            raise ValueError(f"unsupported final-target transform: {self.mode}")
        if self.mode == "ema":
            for name, value in (
                ("arm_ema", self.arm_ema),
                ("finger_ema", self.finger_ema),
            ):
                if not np.isfinite(value) or not 0.0 <= value < 1.0:
                    raise ValueError(f"{name} must be finite and in [0, 1), got {value}")
        if self.mode == "smooth_slew":
            for name, value in (
                ("arm_slew", self.arm_slew),
                ("finger_slew", self.finger_slew),
            ):
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and positive, got {value}")


@dataclass(frozen=True)
class ResidualJointPositionActionConfig:
    """Independent arm/finger residual semantics for scalar position drives."""

    side_order: tuple[str, ...] = ("right", "left")
    rate_semantics: Literal["mapped_residual", "processed_target"] = "mapped_residual"
    arm_scale: float = 0.15
    finger_scale: float = 0.15
    arm_clip: float = 1.0
    finger_clip: float = 1.0
    ema: float = 0.3
    clip_to_joint_limits: bool = True
    delay: TargetDelayConfig = field(default_factory=TargetDelayConfig)
    final_target: FinalTargetTransformConfig = field(default_factory=FinalTargetTransformConfig)
    input_mode: Literal["raw", "normalized"] = "raw"
    normalized_mapping: Literal["linear", "rational"] = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "side_order", tuple(self.side_order))
        if self.rate_semantics not in ("mapped_residual", "processed_target"):
            raise ValueError(f"unsupported action-rate semantics: {self.rate_semantics}")

    def build(self, scene: Scene, device=None) -> "ResidualJointPositionAction":
        return ResidualJointPositionAction.build(scene, config=self, device=device)


@wp.kernel
def process_residual_joint_position_action(
    action: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    reference_num_frames: int,
    reference_num_joint_dof: int,
    reference_joint_target: wp.array(dtype=wp.float32),
    world_num_joint_dof: int,
    action_dim: int,
    joint_dof_ids: wp.array(dtype=wp.int32),
    input_mapping: int,
    input_scale: wp.array(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    clip: wp.array(dtype=wp.float32),
    ema: float,
    final_target_mode: int,
    final_target_ema: wp.array(dtype=wp.float32),
    final_target_slew: wp.array(dtype=wp.float32),
    clip_to_joint_limits: bool,
    rate_on_processed_target: bool,
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    raw_action: wp.array(dtype=wp.float32),
    filtered_action: wp.array(dtype=wp.float32),
    processed_target: wp.array(dtype=wp.float32),
    rate_initialized: wp.array(dtype=wp.int32),
    joint_target: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    action_rate_l2: wp.array(dtype=wp.float32),
) -> None:
    """Map one world's residual and gather its scalar reference target."""
    world = wp.tid()
    action_base = world * action_dim
    joint_base = world * world_num_joint_dof
    frame = timestep[world]
    if frame < 0:
        frame = 0
    if frame >= reference_num_frames:
        frame = reference_num_frames - 1
    reference_base = frame * reference_num_joint_dof
    l2 = float(0.0)
    rate_l2 = float(0.0)
    initialized = rate_initialized[world] != 0

    for joint in range(action_dim):
        index = action_base + joint
        value = map_residual_input(action[index], input_mapping, input_scale[joint])
        residual_delta = value - raw_action[index]
        l2 += value * value
        raw_action[index] = value
        filtered = filter_residual(
            value,
            filtered_action[index],
            scale[joint],
            clip[joint],
            ema,
        )
        filtered_action[index] = filtered

        dof_id = joint_dof_ids[joint]
        target = reference_joint_target[reference_base + dof_id] + filtered
        if clip_to_joint_limits:
            target = wp.clamp(
                target,
                joint_limit_lower[joint_base + dof_id],
                joint_limit_upper[joint_base + dof_id],
            )
        previous_target = processed_target[index] if initialized else reference_joint_target[reference_base + dof_id]
        if final_target_mode == 1:
            previous_weight = final_target_ema[joint]
            target = previous_weight * previous_target + (1.0 - previous_weight) * target
        elif final_target_mode == 2:
            slew = final_target_slew[joint]
            target_delta = target - previous_target
            target = previous_target + slew * wp.tanh(target_delta / slew)
        if rate_on_processed_target:
            if initialized:
                target_delta = target - processed_target[index]
                rate_l2 += target_delta * target_delta
        else:
            rate_l2 += residual_delta * residual_delta
        processed_target[index] = target
        joint_target[joint_base + dof_id] = target

    rate_initialized[world] = 1
    action_l2[world] = l2
    action_rate_l2[world] = rate_l2


@wp.kernel
def reset_residual_joint_position_action(
    reset_mask: wp.array(dtype=wp.int32),
    action_dim: int,
    world_num_joint_dof: int,
    joint_dof_ids: wp.array(dtype=wp.int32),
    raw_action: wp.array(dtype=wp.float32),
    filtered_action: wp.array(dtype=wp.float32),
    processed_target: wp.array(dtype=wp.float32),
    rate_initialized: wp.array(dtype=wp.int32),
    joint_target: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    action_rate_l2: wp.array(dtype=wp.float32),
) -> None:
    """Clear persistent joint-action state for selected worlds."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    action_base = world * action_dim
    joint_base = world * world_num_joint_dof
    rate_initialized[world] = 0
    for joint in range(action_dim):
        index = action_base + joint
        raw_action[index] = 0.0
        filtered_action[index] = 0.0
        processed_target[index] = 0.0
        joint_target[joint_base + joint_dof_ids[joint]] = 0.0
    action_l2[world] = 0.0
    action_rate_l2[world] = 0.0


@dataclass
class ResidualJointPositionAction:
    """Fused reference-relative policy action for implicit scalar position drives."""

    selection: ScalarJointSelection
    reference: DeviceRobotReference
    world_count: int
    world_num_joint_dof: int
    joint_dof_ids: wp.array
    input_mode: Literal["raw", "normalized"]
    normalized_mapping: Literal["linear", "rational"]
    input_mapping: Literal["identity", "linear", "rational"]
    input_mapping_code: int
    input_scale_values: tuple[float, ...]
    input_scale: wp.array
    scale: wp.array
    clip: wp.array
    ema: float
    final_target_mode: Literal["none", "ema", "smooth_slew"]
    final_target_mode_code: int
    final_target_ema: wp.array
    final_target_slew: wp.array
    clip_to_joint_limits: bool
    rate_semantics: Literal["mapped_residual", "processed_target"]
    joint_limit_lower: wp.array
    joint_limit_upper: wp.array
    raw_action: wp.array
    filtered_action: wp.array
    processed_target: wp.array
    rate_initialized: wp.array
    joint_target: wp.array
    action_l2: wp.array
    action_rate_l2: wp.array
    delay: TargetDelayBuffer | None

    @classmethod
    def build(
        cls,
        scene: Scene,
        config: ResidualJointPositionActionConfig | None = None,
        device=None,
    ) -> "ResidualJointPositionAction":
        config = config or ResidualJointPositionActionConfig()
        if len(config.side_order) != len(scene.layout.sides) or set(config.side_order) != set(scene.layout.sides):
            raise ValueError(
                f"joint action side order {config.side_order} must be an exact permutation of "
                f"embodiment sides {scene.layout.sides}"
            )
        selection = scene.layout.select_scalar_joints(
            config.side_order,
            ("arm", "finger"),
            require_names=True,
        )
        if tuple(sorted(selection.q_ids)) != tuple(range(scene.layout.num_joint_q)):
            raise ValueError("residual joint position action must cover every robot q coordinate exactly once")
        if tuple(sorted(selection.dof_ids)) != tuple(range(scene.layout.num_joint_dof)):
            raise ValueError("residual joint position action must cover every robot DOF exactly once")

        scale = np.empty(len(selection.q_ids), dtype=np.float32)
        clip = np.empty(len(selection.q_ids), dtype=np.float32)
        final_target_ema = np.zeros(len(selection.q_ids), dtype=np.float32)
        final_target_slew = np.ones(len(selection.q_ids), dtype=np.float32)
        for block in selection.blocks:
            scale[block.start : block.stop] = config.arm_scale if block.group == "arm" else config.finger_scale
            clip[block.start : block.stop] = config.arm_clip if block.group == "arm" else config.finger_clip
            final_target_ema[block.start : block.stop] = (
                config.final_target.arm_ema if block.group == "arm" else config.final_target.finger_ema
            )
            final_target_slew[block.start : block.stop] = (
                config.final_target.arm_slew if block.group == "arm" else config.final_target.finger_slew
            )
        mapping = build_residual_mapping(
            scale,
            clip,
            ema=config.ema,
            input_mode=config.input_mode,
            normalized_mapping=config.normalized_mapping,
        )

        world_num_joint_dof = scene.model.joint_dof_count // scene.world_count
        target_mode = np.asarray(scene.model.joint_target_mode.numpy(), dtype=np.int32)
        if target_mode.shape != (scene.model.joint_dof_count,):
            raise ValueError(
                f"model joint target mode has shape {target_mode.shape}; expected ({scene.model.joint_dof_count},)"
            )
        target_mode = target_mode.reshape(scene.world_count, world_num_joint_dof)
        allowed_modes = {
            int(newton.JointTargetMode.POSITION),
            int(newton.JointTargetMode.POSITION_VELOCITY),
        }
        invalid_modes = [
            (world, selection.names[joint], int(target_mode[world, dof_id]))
            for world in range(scene.world_count)
            for joint, dof_id in enumerate(selection.dof_ids)
            if int(target_mode[world, dof_id]) not in allowed_modes
        ]
        if invalid_modes:
            raise ValueError(
                "residual joint position action requires position-capable target modes; "
                f"got (world, joint, mode) {invalid_modes}"
            )
        reference = DeviceRobotReference.build(
            scene.robot_reference,
            scene.layout,
            sides=config.side_order,
            device=device,
        )
        delay = None
        arm_joints = scene.layout.select_scalar_joints(
            config.side_order,
            ("arm",),
            require_names=True,
        )
        if config.delay.max_steps > 0:
            if not arm_joints.dof_ids:
                raise ValueError("arm target delay requires at least one articulated arm joint")
            delay = TargetDelayBuffer.build(
                world_count=scene.world_count,
                num_joint_dof=world_num_joint_dof,
                delayed_dof_ids=arm_joints.dof_ids,
                config=config.delay,
                device=device,
            )

        action_dim = len(selection.q_ids)
        return cls(
            selection=selection,
            reference=reference,
            world_count=scene.world_count,
            world_num_joint_dof=world_num_joint_dof,
            joint_dof_ids=wp.array(selection.dof_ids, dtype=wp.int32, device=device),
            input_mode=mapping.input_mode,
            normalized_mapping=mapping.normalized_mapping,
            input_mapping=mapping.input_mapping,
            input_mapping_code=mapping.input_mapping_code,
            input_scale_values=tuple(float(value) for value in mapping.input_scale),
            input_scale=wp.array(mapping.input_scale, dtype=wp.float32, device=device),
            scale=wp.array(mapping.scale, dtype=wp.float32, device=device),
            clip=wp.array(mapping.clip, dtype=wp.float32, device=device),
            ema=mapping.ema,
            final_target_mode=config.final_target.mode,
            final_target_mode_code={"none": 0, "ema": 1, "smooth_slew": 2}[config.final_target.mode],
            final_target_ema=wp.array(final_target_ema, dtype=wp.float32, device=device),
            final_target_slew=wp.array(final_target_slew, dtype=wp.float32, device=device),
            clip_to_joint_limits=config.clip_to_joint_limits,
            rate_semantics=config.rate_semantics,
            joint_limit_lower=scene.model.joint_limit_lower,
            joint_limit_upper=scene.model.joint_limit_upper,
            raw_action=wp.zeros(scene.world_count * action_dim, dtype=wp.float32, device=device),
            filtered_action=wp.zeros(scene.world_count * action_dim, dtype=wp.float32, device=device),
            processed_target=wp.zeros(scene.world_count * action_dim, dtype=wp.float32, device=device),
            rate_initialized=wp.zeros(scene.world_count, dtype=wp.int32, device=device),
            joint_target=wp.zeros(scene.world_count * world_num_joint_dof, dtype=wp.float32, device=device),
            action_l2=wp.zeros(scene.world_count, dtype=wp.float32, device=device),
            action_rate_l2=wp.zeros(scene.world_count, dtype=wp.float32, device=device),
            delay=delay,
        )

    @property
    def sides(self) -> tuple[str, ...]:
        return self.selection.sides

    @property
    def action_dim(self) -> int:
        return len(self.selection.q_ids)

    @property
    def processed_dim(self) -> int:
        return self.action_dim

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(
            f"{block.side}_{block.group}_joint_residual[{','.join(self.selection.names[block.start : block.stop])}]"
            for block in self.selection.blocks
            if block.count > 0
        )

    @property
    def block_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple((block.start, block.stop) for block in self.selection.blocks if block.count > 0)

    def process(self, action: wp.array, timestep: wp.array, state=None) -> None:
        wp.launch(
            process_residual_joint_position_action,
            dim=self.world_count,
            inputs=[
                action,
                timestep,
                self.reference.num_frames,
                self.reference.num_joint_dof,
                self.reference.joint_target,
                self.world_num_joint_dof,
                self.action_dim,
                self.joint_dof_ids,
                self.input_mapping_code,
                self.input_scale,
                self.scale,
                self.clip,
                self.ema,
                self.final_target_mode_code,
                self.final_target_ema,
                self.final_target_slew,
                self.clip_to_joint_limits,
                self.rate_semantics == "processed_target",
                self.joint_limit_lower,
                self.joint_limit_upper,
            ],
            outputs=[
                self.raw_action,
                self.filtered_action,
                self.processed_target,
                self.rate_initialized,
                self.joint_target,
                self.action_l2,
                self.action_rate_l2,
            ],
        )

    def prepare_control(self, control) -> None:
        if self.delay is None:
            wp.copy(control.joint_target_pos, self.joint_target)

    def apply_control(self, state, control) -> None:
        if self.delay is not None:
            wp.copy(control.joint_target_pos, self.delay.advance(self.joint_target))

    def reset(self, reset_mask: wp.array) -> None:
        wp.launch(
            reset_residual_joint_position_action,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.action_dim,
                self.world_num_joint_dof,
                self.joint_dof_ids,
            ],
            outputs=[
                self.raw_action,
                self.filtered_action,
                self.processed_target,
                self.rate_initialized,
                self.joint_target,
                self.action_l2,
                self.action_rate_l2,
            ],
        )
        if self.delay is not None:
            self.delay.reset(reset_mask)
