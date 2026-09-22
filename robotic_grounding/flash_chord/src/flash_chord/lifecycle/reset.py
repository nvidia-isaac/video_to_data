# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-conditioned reset strategies and their shared runtime protocols."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import newton
import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.embodiments.binding import DeviceRobotReference
from flash_chord.embodiments.frames import DeviceBodyFrameMap
from flash_chord.scene.builder import Scene
from flash_chord.utils.quat import quat_geodesic_angle, quat_mul_xyzw, wxyz_to_xyzw

RESET_CONTEXT_NAMES = (
    "applied_voc_scale",
    "target_voc_scale",
    "normalized_settling_progress",
)


def _fill_frame_object_q(scene: Scene, reference: Reference, frame_id: int, row: np.ndarray) -> None:
    """Fill object free-joint and articulation coordinates for one reference frame."""
    obj_pos = np.asarray(reference.object_body_pos_w()[frame_id])  # (B, 3)
    obj_quat = np.asarray(reference.object_body_quat_w()[frame_id])  # (B, 4) wxyz
    articulation = np.asarray(reference.object_articulation()[frame_id])  # (A,)
    for binding in scene.objects:
        root = binding.root
        q_ids = root.free_q_ids
        row[list(q_ids[:3])] = obj_pos[root.reference_body_id]
        row[list(q_ids[3:])] = wxyz_to_xyzw(obj_quat[root.reference_body_id])
        for joint in binding.articulations:
            if joint.reference_id >= articulation.shape[0]:
                raise ValueError(
                    f"scene object {binding.name!r} maps articulation column {joint.reference_id}, "
                    f"but the reference has {articulation.shape[0]} columns"
                )
            row[joint.q_id] = float(articulation[joint.reference_id])


def reset_scene_to_frame(scene: Scene, reference: Reference, frame_id: int, state) -> None:
    """Host one-shot: snap ``state`` (joint_q + body transforms) to ``reference`` at ``frame_id`` for every
    world. For the runtime per-step reset use the device path (:class:`ReferenceResetTable` + :func:`reset_worlds`)."""
    model = scene.model
    per_world_q = model.joint_coord_count // scene.world_count
    joint_q = np.array(model.joint_q.numpy(), dtype=np.float64)  # build defaults
    row = joint_q[:per_world_q].copy()
    row[: scene.robot_reference.num_joint_q] = scene.robot_reference.joint_q[frame_id]
    _fill_frame_object_q(scene, reference, frame_id, row)
    joint_q[:] = np.tile(row, scene.world_count)
    for world in range(scene.world_count):
        q_base = world * per_world_q
        for object_id, binding in enumerate(scene.objects):
            root_q = q_base + binding.root.free_q_ids[0]
            joint_q[root_q : root_q + 3] += scene.object_root_position_offsets_w[world, object_id]
    state.joint_q.assign(joint_q.astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def build_reference_joint_q(scene: Scene, reference: Reference) -> np.ndarray:
    """Per-world ``joint_q`` for every reference frame, ``[num_frames, per_world_q]`` (host, setup-time)."""
    per_world_q = scene.model.joint_coord_count // scene.world_count
    default_row = np.array(scene.model.joint_q.numpy(), dtype=np.float32)[:per_world_q]
    table = np.tile(default_row, (reference.num_frames, 1))
    table[:, : scene.robot_reference.num_joint_q] = scene.robot_reference.joint_q
    for f in range(reference.num_frames):
        _fill_frame_object_q(scene, reference, f, table[f])
    return table


@wp.kernel
def reset_kernel(
    joint_q: wp.array(dtype=wp.float32),  # [W*per_world_q] in/out
    joint_qd: wp.array(dtype=wp.float32),  # [W*per_world_dof] in/out
    per_world_q: int,
    per_world_dof: int,
    reset_mask: wp.array(dtype=wp.int32),  # [W] nonzero resets that world
    reset_frame: wp.array(dtype=wp.int32),  # [W] reference frame each world resets to
    ref_joint_q: wp.array(dtype=wp.float32, ndim=2),  # [F, per_world_q]
    reference_joint_q_offset: wp.array(dtype=wp.float32),  # [W*per_world_q]
    finger_q_ids: wp.array(dtype=wp.int32),
    finger_lower: wp.array(dtype=wp.float32),
    finger_upper: wp.array(dtype=wp.float32),
    num_fingers: int,
    finger_scale: wp.array(dtype=wp.float32),
    object_root_q_ids: wp.array(dtype=wp.int32),
    object_root_position_offsets_w: wp.array(dtype=wp.vec3),
    num_objects: int,
) -> None:
    """Reset selected worlds to a reference pose, optional open fingers, and zero velocity."""
    w = wp.tid()
    if reset_mask[w] == 0:
        return
    f = reset_frame[w]
    base = w * per_world_q
    for c in range(per_world_q):
        joint_q[base + c] = ref_joint_q[f, c] + reference_joint_q_offset[base + c]
    for object_id in range(num_objects):
        root_q = base + object_root_q_ids[object_id]
        offset = object_root_position_offsets_w[w * num_objects + object_id]
        joint_q[root_q] = joint_q[root_q] + offset[0]
        joint_q[root_q + 1] = joint_q[root_q + 1] + offset[1]
        joint_q[root_q + 2] = joint_q[root_q + 2] + offset[2]
    openness = finger_scale[w]
    if openness < 1.0:
        for finger in range(num_fingers):
            q_id = finger_q_ids[finger]
            joint_q[base + q_id] = wp.clamp(
                openness * ref_joint_q[f, q_id],
                finger_lower[finger],
                finger_upper[finger],
            )
    dof_base = w * per_world_dof
    for dof in range(per_world_dof):
        joint_qd[dof_base + dof] = 0.0


@dataclass
class ReferenceResetTable:
    """Device table of per-frame reference ``joint_q`` (+ a scratch per-world frame index) for the reset [WK]."""

    per_world_q: int
    per_world_dof: int
    world_count: int
    num_frames: int
    ref_joint_q: wp.array  # float32 [F, per_world_q]
    finger_q_ids: wp.array  # int32 [num_fingers]
    finger_lower: wp.array  # float32 [num_fingers]
    finger_upper: wp.array  # float32 [num_fingers]
    exact_finger_scale: wp.array  # float32 [W], constant ones
    zero_joint_q_offset: wp.array  # float32 [W*per_world_q], constant zeros
    object_root_q_ids: wp.array  # int32 [O]
    object_root_position_offsets_w: wp.array  # vec3 [W*O]
    num_objects: int
    reset_frame: wp.array  # int32 [W] scratch
    all_reset_mask: wp.array  # int32 [W], constant ones

    @classmethod
    def build(cls, scene: Scene, reference: Reference, device=None) -> "ReferenceResetTable":
        table = build_reference_joint_q(scene, reference)
        finger_q_ids = tuple(q_id for hand in scene.layout.hands for q_id in hand.finger_q_ids)
        finger_dof_ids = tuple(dof_id for hand in scene.layout.hands for dof_id in hand.finger_dof_ids)
        joint_lower = np.asarray(scene.model.joint_limit_lower.numpy(), dtype=np.float32)
        joint_upper = np.asarray(scene.model.joint_limit_upper.numpy(), dtype=np.float32)
        return cls(
            per_world_q=table.shape[1],
            per_world_dof=scene.model.joint_dof_count // scene.world_count,
            world_count=scene.world_count,
            num_frames=table.shape[0],
            ref_joint_q=wp.array(table, dtype=wp.float32, device=device),
            finger_q_ids=wp.array(finger_q_ids, dtype=wp.int32, device=device),
            finger_lower=wp.array(joint_lower[list(finger_dof_ids)], dtype=wp.float32, device=device),
            finger_upper=wp.array(joint_upper[list(finger_dof_ids)], dtype=wp.float32, device=device),
            exact_finger_scale=wp.ones(scene.world_count, dtype=wp.float32, device=device),
            zero_joint_q_offset=wp.zeros(scene.world_count * table.shape[1], dtype=wp.float32, device=device),
            object_root_q_ids=wp.array(
                [binding.root.free_q_ids[0] for binding in scene.objects],
                dtype=wp.int32,
                device=device,
            ),
            object_root_position_offsets_w=wp.array(
                scene.object_root_position_offsets_w.reshape(-1, 3),
                dtype=wp.vec3,
                device=device,
            ),
            num_objects=len(scene.objects),
            reset_frame=wp.zeros(scene.world_count, dtype=wp.int32, device=device),
            all_reset_mask=wp.ones(scene.world_count, dtype=wp.int32, device=device),
        )


def reset_worlds(
    model,
    table: ReferenceResetTable,
    state,
    frame_id: int | None = None,
    reset_frame: wp.array | None = None,
    reset_mask: wp.array | None = None,
    finger_scale: wp.array | None = None,
    reference_joint_q_offset: wp.array | None = None,
) -> None:
    """Scatter the reference ``joint_q`` into ``state`` (device [WK]) then refresh body transforms via
    ``eval_fk``. Pass ``frame_id`` to reset all worlds to one frame, or ``reset_frame`` (a ``[W]`` device
    int array) to reset each world to its own frame (auto-reset of a done subset)."""
    if reset_frame is None:
        if frame_id is None:
            raise ValueError("pass frame_id or reset_frame")
        table.reset_frame.assign(np.full(table.world_count, int(frame_id), dtype=np.int32))
        reset_frame = table.reset_frame
    if reset_mask is None:
        reset_mask = table.all_reset_mask
    if finger_scale is None:
        finger_scale = table.exact_finger_scale
    if reference_joint_q_offset is None:
        reference_joint_q_offset = table.zero_joint_q_offset
    wp.launch(
        reset_kernel,
        dim=table.world_count,
        inputs=[
            state.joint_q,
            state.joint_qd,
            table.per_world_q,
            table.per_world_dof,
            reset_mask,
            reset_frame,
            table.ref_joint_q,
            reference_joint_q_offset,
            table.finger_q_ids,
            table.finger_lower,
            table.finger_upper,
            table.finger_q_ids.shape[0],
            finger_scale,
            table.object_root_q_ids,
            table.object_root_position_offsets_w,
            table.num_objects,
        ],
    )
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


@dataclass(frozen=True)
class ReferenceResetConfig:
    """Random reference reset with finger opening and a configurable VOC settling phase."""

    reset_finger_openness: float = 0.7
    always_reset_to_first_frame: bool = False
    reset_to_first_frame_probability: float = 0.1
    first_frame_voc_threshold: float = 0.1
    voc_decay_steps: int = 20
    voc_decay_mode: Literal["step", "linear"] = "step"
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.reset_finger_openness <= 1.0:
            raise ValueError(f"reset_finger_openness must be in [0, 1], got {self.reset_finger_openness}")
        if not 0.0 <= self.reset_to_first_frame_probability <= 1.0:
            raise ValueError(
                f"reset_to_first_frame_probability must be in [0, 1], got {self.reset_to_first_frame_probability}"
            )
        if self.first_frame_voc_threshold < 0.0:
            raise ValueError(f"first_frame_voc_threshold must be non-negative, got {self.first_frame_voc_threshold}")
        if self.voc_decay_steps < 0:
            raise ValueError(f"voc_decay_steps must be non-negative, got {self.voc_decay_steps}")
        if self.voc_decay_mode not in ("step", "linear"):
            raise ValueError(f"voc_decay_mode must be 'step' or 'linear', got {self.voc_decay_mode!r}")

    def build(self, world_count: int, num_frames: int, device=None) -> ResetPolicy:
        """Build the reference reset policy."""
        return ReferenceReset.build(world_count, num_frames, config=self, device=device)


@runtime_checkable
class ResetPolicy(Protocol):
    """Device reset state consumed directly by an RL environment."""

    reset_frame: wp.array
    finger_scale: wp.array
    applied_voc_scale: wp.array

    def sample(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        """Sample reset state for selected worlds."""
        ...

    def prepare_explicit(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        """Prepare an exact caller-selected reset without a settling phase."""
        ...

    def advance(
        self,
        terminated: wp.array,
        truncated: wp.array,
        timestep: wp.array,
        episode_step: wp.array,
        curriculum_voc_scale: wp.array,
    ) -> None:
        """Advance settling and reference counters after one transition."""
        ...

    def sync_curriculum(self, curriculum_voc_scale: wp.array) -> None:
        """Apply a changed curriculum scale to worlds past settling."""
        ...


@runtime_checkable
class ResetContext(Protocol):
    """Optional capability for publishing the critic's per-world reset/VOC context."""

    context_names: tuple[str, ...]

    def compute_context(self, curriculum_voc_scale: wp.array, output: wp.array) -> wp.array:
        """Write packed float32 context rows to ``output`` and return it."""
        ...


@runtime_checkable
class FirstFrameResetCurriculum(Protocol):
    """Optional capability for changing reference-frame sampling between curriculum stages."""

    def set_reset_to_first_frame_probability(self, probability: float | None) -> float:
        """Set the stage override, or restore the reset configuration with ``None``; return the resolved value."""
        ...


@runtime_checkable
class ImmediateFirstFrameResetCurriculum(Protocol):
    """Optional capability for mixing deployment-exact frame-zero resets into training."""

    def set_immediate_first_frame_probability(self, probability: float | None) -> float:
        """Set the immediate-start override, or restore the reset configuration with ``None``."""
        ...


@runtime_checkable
class ReferenceOffsetReset(Protocol):
    """Reset policy that writes settling offsets into an environment-owned buffer."""

    def bind_reference_joint_q_offset(
        self,
        scene: Scene,
        reference: Reference,
        output: wp.array,
    ) -> None:
        """Bind scene mappings, reference values, and the persistent offset output."""
        ...


@runtime_checkable
class ResetSpec(Protocol):
    """Setup-time reset-policy builder selected by configuration."""

    def build(self, world_count: int, num_frames: int, device=None) -> ResetPolicy:
        """Bind runtime dimensions and return a reset policy."""
        ...


@wp.kernel
def sample_reference_reset(
    reset_mask: wp.array(dtype=wp.int32),
    num_frames: int,
    voc_decay_steps: int,
    always_first_frame: int,
    first_frame_probability: wp.array(dtype=wp.float32),
    first_frame_voc_threshold: float,
    reset_finger_openness: float,
    seed: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    reset_count: wp.array(dtype=wp.int32),
    reset_frame: wp.array(dtype=wp.int32),
    finger_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
) -> None:
    """Sample one reference reset frame and finger factor for each selected world."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return

    offset = world + reset_count[world] * 104729
    rng = wp.rand_init(seed, offset)
    frame = int(0)
    random_frame_count = num_frames - 1
    if always_first_frame == 0 and random_frame_count > 0:
        frame = int(wp.randf(rng) * float(random_frame_count))
        if curriculum_voc_scale[0] < first_frame_voc_threshold:
            if wp.randf(rng) < first_frame_probability[0]:
                frame = 0

    reset_frame[world] = frame
    finger_scale[world] = wp.randf(rng) * reset_finger_openness
    steps_since_reset[world] = 0
    if voc_decay_steps == 0:
        applied_voc_scale[world] = curriculum_voc_scale[0]
    else:
        applied_voc_scale[world] = 1.0
    reset_count[world] += 1


@wp.kernel
def prepare_explicit_reset(
    reset_mask: wp.array(dtype=wp.int32),
    voc_decay_steps: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    finger_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
) -> None:
    """Preserve the exact-pose semantics of an explicitly indexed reset."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    finger_scale[world] = 1.0
    steps_since_reset[world] = voc_decay_steps
    applied_voc_scale[world] = curriculum_voc_scale[0]


@wp.kernel
def advance_reference_reset(
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    voc_decay_steps: int,
    linear_decay: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
) -> None:
    """Hold the reset frame with VOC=1, then advance under the curriculum scale."""
    world = wp.tid()
    if terminated[world] != 0 or truncated[world] != 0:
        return

    age = steps_since_reset[world] + 1
    steps_since_reset[world] = age
    episode_step[world] += 1
    if voc_decay_steps == 0 or age >= voc_decay_steps:
        applied_voc_scale[world] = curriculum_voc_scale[0]
        timestep[world] += 1
    elif linear_decay != 0:
        progress = float(age) / float(voc_decay_steps)
        applied_voc_scale[world] = wp.max(
            0.0,
            1.0 + (curriculum_voc_scale[0] - 1.0) * progress,
        )
    else:
        applied_voc_scale[world] = 1.0


@wp.kernel
def sync_reference_voc(
    voc_decay_steps: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
) -> None:
    """Update settled worlds immediately when the global curriculum changes."""
    world = wp.tid()
    if steps_since_reset[world] >= voc_decay_steps:
        applied_voc_scale[world] = curriculum_voc_scale[0]


@wp.kernel
def publish_reset_context(
    applied_voc_scale: wp.array(dtype=wp.float32),
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    voc_decay_steps: int,
    context: wp.array(dtype=wp.float32),
) -> None:
    """Publish applied/target VOC and normalized settling progress for one world."""
    world = wp.tid()
    progress = float(1.0)
    if voc_decay_steps > 0:
        progress = wp.clamp(float(steps_since_reset[world]) / float(voc_decay_steps), 0.0, 1.0)
    output = world * 3
    context[output] = applied_voc_scale[world]
    context[output + 1] = curriculum_voc_scale[0]
    context[output + 2] = progress


@dataclass
class ReferenceReset:
    """Persistent per-world reference reset state with graph-capturable operations."""

    config: ReferenceResetConfig
    world_count: int
    num_frames: int
    reset_frame: wp.array
    finger_scale: wp.array
    steps_since_reset: wp.array
    applied_voc_scale: wp.array
    reset_count: wp.array
    first_frame_probability: wp.array

    context_names = RESET_CONTEXT_NAMES

    def compute_context(self, curriculum_voc_scale: wp.array, output: wp.array) -> wp.array:
        """Publish the current critic-only reset context into a caller buffer."""
        expected = (self.world_count * len(self.context_names),)
        if output.shape != expected:
            raise ValueError(f"reset context output has shape {output.shape}; expected {expected}")
        if output.dtype != wp.float32:
            raise TypeError(f"reset context output must be float32, got {output.dtype}")
        if output.device != self.applied_voc_scale.device:
            raise ValueError(
                f"reset context output is on {output.device}; expected reset device {self.applied_voc_scale.device}"
            )
        wp.launch(
            publish_reset_context,
            dim=self.world_count,
            inputs=[
                self.applied_voc_scale,
                curriculum_voc_scale,
                self.steps_since_reset,
                self.config.voc_decay_steps,
            ],
            outputs=[output],
        )
        return output

    @classmethod
    def build(
        cls,
        world_count: int,
        num_frames: int,
        config: ReferenceResetConfig | None = None,
        device=None,
    ) -> "ReferenceReset":
        config = config or ReferenceResetConfig()
        return cls(
            config=config,
            world_count=world_count,
            num_frames=num_frames,
            reset_frame=wp.zeros(world_count, dtype=wp.int32, device=device),
            finger_scale=wp.ones(world_count, dtype=wp.float32, device=device),
            steps_since_reset=wp.zeros(world_count, dtype=wp.int32, device=device),
            applied_voc_scale=wp.ones(world_count, dtype=wp.float32, device=device),
            reset_count=wp.zeros(world_count, dtype=wp.int32, device=device),
            first_frame_probability=wp.array(
                [config.reset_to_first_frame_probability],
                dtype=wp.float32,
                device=device,
            ),
        )

    def set_reset_to_first_frame_probability(self, probability: float | None) -> float:
        """Update the graph-captured sampling scalar without rebuilding or recapturing the reset kernel."""
        resolved = self.config.reset_to_first_frame_probability if probability is None else probability
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError(
                f"reset_to_first_frame_probability must be None or finite and in [0, 1], got {probability}"
            )
        self.first_frame_probability.assign(np.asarray([resolved], dtype=np.float32))
        return float(resolved)

    def sample(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        wp.launch(
            sample_reference_reset,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.num_frames,
                self.config.voc_decay_steps,
                int(self.config.always_reset_to_first_frame),
                self.first_frame_probability,
                self.config.first_frame_voc_threshold,
                self.config.reset_finger_openness,
                self.config.seed,
                curriculum_voc_scale,
            ],
            outputs=[
                self.reset_count,
                self.reset_frame,
                self.finger_scale,
                self.steps_since_reset,
                self.applied_voc_scale,
            ],
        )

    def prepare_explicit(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        wp.launch(
            prepare_explicit_reset,
            dim=self.world_count,
            inputs=[reset_mask, self.config.voc_decay_steps, curriculum_voc_scale],
            outputs=[self.finger_scale, self.steps_since_reset, self.applied_voc_scale],
        )

    def advance(
        self,
        terminated: wp.array,
        truncated: wp.array,
        timestep: wp.array,
        episode_step: wp.array,
        curriculum_voc_scale: wp.array,
    ) -> None:
        wp.launch(
            advance_reference_reset,
            dim=self.world_count,
            inputs=[
                terminated,
                truncated,
                self.config.voc_decay_steps,
                int(self.config.voc_decay_mode == "linear"),
                curriculum_voc_scale,
            ],
            outputs=[self.steps_since_reset, self.applied_voc_scale, timestep, episode_step],
        )

    def sync_curriculum(self, curriculum_voc_scale: wp.array) -> None:
        wp.launch(
            sync_reference_voc,
            dim=self.world_count,
            inputs=[self.config.voc_decay_steps, curriculum_voc_scale, self.steps_since_reset],
            outputs=[self.applied_voc_scale],
        )


# ReconBody reset strategy.


@dataclass(frozen=True)
class ReconBodyResetConfig:
    """Random reference reset and source-faithful 50-step settling timeline."""

    reset_freeze_steps: int = 50
    voc_decay_steps: int = 10
    reset_voc_scale: float = 1.0
    shoulder_spread: float = 0.2
    start_frame: int = 0
    end_frame: int = -1
    always_reset_to_first_frame: bool = False
    reset_to_first_frame_probability: float = 0.0
    immediate_first_frame_probability: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.reset_freeze_steps < 0 or self.voc_decay_steps < 0:
            raise ValueError("ReconBody reset freeze and VOC decay steps must be non-negative")
        if self.voc_decay_steps > self.reset_freeze_steps:
            raise ValueError("ReconBody VOC decay cannot exceed the reset freeze")
        if not math.isfinite(self.reset_voc_scale) or self.reset_voc_scale < 0.0:
            raise ValueError("ReconBody reset VOC scale must be finite and non-negative")
        if not math.isfinite(self.shoulder_spread) or self.shoulder_spread < 0.0:
            raise ValueError("ReconBody shoulder spread must be finite and non-negative")
        if self.start_frame < 0 or self.end_frame < -1:
            raise ValueError("ReconBody reset frame bounds must be non-negative, or -1 for the final frame")
        if not 0.0 <= self.reset_to_first_frame_probability <= 1.0:
            raise ValueError("ReconBody first-frame reset probability must be in [0, 1]")
        if not 0.0 <= self.immediate_first_frame_probability <= 1.0:
            raise ValueError("ReconBody immediate first-frame probability must be in [0, 1]")

    def build(self, world_count: int, num_frames: int, device=None) -> ReconBodyReset:
        return ReconBodyReset.build(world_count, num_frames, config=self, device=device)


@wp.kernel
def sample_recon_body_reset(
    reset_mask: wp.array(dtype=wp.int32),
    reference_joint_q: wp.array(dtype=wp.float32, ndim=2),
    q_per_world: int,
    start_frame: int,
    end_frame: int,
    always_first_frame: int,
    first_frame_probability: wp.array(dtype=wp.float32),
    immediate_first_frame_probability: wp.array(dtype=wp.float32),
    freeze_steps: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    reset_voc_scale: float,
    shoulder_spread: float,
    left_shoulder_q_id: int,
    right_shoulder_q_id: int,
    finger_q_ids: wp.array(dtype=wp.int32),
    seed: int,
    reset_count: wp.array(dtype=wp.int32),
    reset_frame: wp.array(dtype=wp.int32),
    finger_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    raw_joint_q_offset: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    if reset_mask[world] == 0:
        return

    rng = wp.rand_init(seed, world + reset_count[world] * 104729)
    immediate = int(0)  # noqa: RUF046, UP018 - Warp requires an explicitly typed mutable local.
    if immediate_first_frame_probability[0] > 0.0 and wp.randf(rng) < immediate_first_frame_probability[0]:
        immediate = 1
    frame = start_frame
    if immediate != 0:
        frame = 0
    frame_count = end_frame - start_frame + 1
    if immediate == 0 and always_first_frame == 0 and frame_count > 1:
        frame = start_frame + int(wp.randf(rng) * float(frame_count))
        if wp.randf(rng) < first_frame_probability[0]:
            frame = 0

    base = world * q_per_world
    for coordinate in range(q_per_world):
        raw_joint_q_offset[base + coordinate] = 0.0
        reference_joint_q_offset[base + coordinate] = 0.0
    if immediate == 0:
        raw_joint_q_offset[base + left_shoulder_q_id] = shoulder_spread
        raw_joint_q_offset[base + right_shoulder_q_id] = -shoulder_spread
        reference_joint_q_offset[base + left_shoulder_q_id] = shoulder_spread
        reference_joint_q_offset[base + right_shoulder_q_id] = -shoulder_spread
        for finger in range(finger_q_ids.shape[0]):
            q_id = finger_q_ids[finger]
            offset = -reference_joint_q[frame, q_id]
            raw_joint_q_offset[base + q_id] = offset
            reference_joint_q_offset[base + q_id] = offset

    reset_frame[world] = frame
    finger_scale[world] = 1.0
    if immediate != 0:
        steps_since_reset[world] = freeze_steps + 1
        applied_voc_scale[world] = curriculum_voc_scale[0]
    else:
        steps_since_reset[world] = 0
        applied_voc_scale[world] = reset_voc_scale
    reset_count[world] += 1


@wp.kernel
def prepare_explicit_recon_body_reset(
    reset_mask: wp.array(dtype=wp.int32),
    q_per_world: int,
    settled_age: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    finger_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    raw_joint_q_offset: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    base = world * q_per_world
    for coordinate in range(q_per_world):
        raw_joint_q_offset[base + coordinate] = 0.0
        reference_joint_q_offset[base + coordinate] = 0.0
    finger_scale[world] = 1.0
    steps_since_reset[world] = settled_age
    applied_voc_scale[world] = curriculum_voc_scale[0]


@wp.kernel
def advance_recon_body_reset(
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    freeze_steps: int,
    offset_decay_steps: int,
    voc_decay_steps: int,
    reset_voc_scale: float,
    q_per_world: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    raw_joint_q_offset: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
) -> None:
    world = wp.tid()
    if terminated[world] != 0 or truncated[world] != 0:
        return

    age = steps_since_reset[world] + 1
    steps_since_reset[world] = age
    episode_step[world] += 1

    offset_scale = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    if offset_decay_steps > 0 and age < offset_decay_steps:
        offset_scale = 1.0 - float(age) / float(offset_decay_steps)
    base = world * q_per_world
    for coordinate in range(q_per_world):
        reference_joint_q_offset[base + coordinate] = offset_scale * raw_joint_q_offset[base + coordinate]

    decay_start = freeze_steps - voc_decay_steps
    if voc_decay_steps == 0 or age >= freeze_steps:
        applied_voc_scale[world] = curriculum_voc_scale[0]
    elif age <= decay_start:
        applied_voc_scale[world] = reset_voc_scale
    else:
        progress = float(age - decay_start) / float(voc_decay_steps)
        applied_voc_scale[world] = reset_voc_scale + progress * (curriculum_voc_scale[0] - reset_voc_scale)

    # Preserve the source's strict `steps_since_reset > freeze_steps` release.
    if age > freeze_steps:
        timestep[world] += 1


@wp.kernel
def sync_recon_body_voc(
    freeze_steps: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    if steps_since_reset[world] >= freeze_steps:
        applied_voc_scale[world] = curriculum_voc_scale[0]


@wp.kernel
def publish_recon_body_reset_context(
    applied_voc_scale: wp.array(dtype=wp.float32),
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    freeze_steps: int,
    context: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    progress = float(1.0)  # noqa: UP018 - Warp requires an explicitly typed mutable local.
    if freeze_steps > 0:
        progress = wp.clamp(float(steps_since_reset[world]) / float(freeze_steps), 0.0, 1.0)
    output = world * 3
    context[output] = applied_voc_scale[world]
    context[output + 1] = curriculum_voc_scale[0]
    context[output + 2] = progress


@dataclass
class ReconBodyReset:
    """Graph-capturable ReconBody reset state, bound once to a G1 scene."""

    config: ReconBodyResetConfig
    world_count: int
    num_frames: int
    reset_frame: wp.array
    finger_scale: wp.array
    steps_since_reset: wp.array
    applied_voc_scale: wp.array
    reset_count: wp.array
    first_frame_probability: wp.array
    immediate_first_frame_probability: wp.array
    raw_joint_q_offset: wp.array
    reference_joint_q_offset: wp.array
    reference_joint_q: wp.array
    finger_q_ids: wp.array
    q_per_world: int = 0
    left_shoulder_q_id: int = -1
    right_shoulder_q_id: int = -1
    start_frame: int = 0
    end_frame: int = -1

    context_names = RESET_CONTEXT_NAMES

    @classmethod
    def build(
        cls,
        world_count: int,
        num_frames: int,
        config: ReconBodyResetConfig | None = None,
        device=None,
    ) -> ReconBodyReset:
        config = config or ReconBodyResetConfig()
        return cls(
            config=config,
            world_count=world_count,
            num_frames=num_frames,
            reset_frame=wp.zeros(world_count, dtype=wp.int32, device=device),
            finger_scale=wp.ones(world_count, dtype=wp.float32, device=device),
            steps_since_reset=wp.zeros(world_count, dtype=wp.int32, device=device),
            applied_voc_scale=wp.ones(world_count, dtype=wp.float32, device=device),
            reset_count=wp.zeros(world_count, dtype=wp.int32, device=device),
            first_frame_probability=wp.array(
                [config.reset_to_first_frame_probability], dtype=wp.float32, device=device
            ),
            immediate_first_frame_probability=wp.array(
                [config.immediate_first_frame_probability], dtype=wp.float32, device=device
            ),
            raw_joint_q_offset=wp.empty(0, dtype=wp.float32, device=device),
            reference_joint_q_offset=wp.empty(0, dtype=wp.float32, device=device),
            reference_joint_q=wp.empty((0, 0), dtype=wp.float32, device=device),
            finger_q_ids=wp.empty(0, dtype=wp.int32, device=device),
        )

    def bind_reference_joint_q_offset(self, scene: Scene, reference: Reference, output: wp.array) -> None:
        scalar = scene.layout.scalar_joints
        if scalar is None:
            raise ValueError("ReconBody reset requires named scalar robot joints")
        q_by_name = dict(zip(scalar.names, scalar.q_ids, strict=True))
        shoulder_names = ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint")
        missing = tuple(name for name in shoulder_names if name not in q_by_name)
        if missing:
            raise ValueError(f"ReconBody reset is missing shoulder joints {missing}")
        finger_q_ids = tuple(q_id for hand in scene.layout.hands for q_id in hand.finger_q_ids)
        q_per_world = scene.model.joint_coord_count // scene.world_count
        expected = (self.world_count * q_per_world,)
        if output.shape != expected or output.dtype != wp.float32:
            raise ValueError(f"ReconBody reference offset must be float32 with shape {expected}")
        end_frame = self.num_frames - 1 if self.config.end_frame == -1 else self.config.end_frame
        if self.config.start_frame > end_frame or end_frame >= self.num_frames:
            raise ValueError(
                f"ReconBody reset frame window [{self.config.start_frame}, {end_frame}] "
                f"is outside {self.num_frames} frames"
            )
        self.q_per_world = q_per_world
        self.left_shoulder_q_id = q_by_name[shoulder_names[0]]
        self.right_shoulder_q_id = q_by_name[shoulder_names[1]]
        self.start_frame = self.config.start_frame
        self.end_frame = end_frame
        self.reference_joint_q_offset = output
        self.raw_joint_q_offset = wp.zeros(expected[0], dtype=wp.float32, device=output.device)
        self.reference_joint_q = wp.array(scene.robot_reference.joint_q, dtype=wp.float32, device=output.device)
        self.finger_q_ids = wp.array(finger_q_ids, dtype=wp.int32, device=output.device)

    def _require_bound(self) -> None:
        if self.q_per_world <= 0:
            raise RuntimeError("ReconBody reset has not been bound to a scene")

    def set_reset_to_first_frame_probability(self, probability: float | None) -> float:
        resolved = self.config.reset_to_first_frame_probability if probability is None else probability
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError("reset_to_first_frame_probability must be None or finite and in [0, 1]")
        self.first_frame_probability.assign(np.asarray([resolved], dtype=np.float32))
        return float(resolved)

    def set_immediate_first_frame_probability(self, probability: float | None) -> float:
        resolved = self.config.immediate_first_frame_probability if probability is None else probability
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError("immediate_first_frame_probability must be None or finite and in [0, 1]")
        self.immediate_first_frame_probability.assign(np.asarray([resolved], dtype=np.float32))
        return float(resolved)

    def sample(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        self._require_bound()
        wp.launch(
            sample_recon_body_reset,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.reference_joint_q,
                self.q_per_world,
                self.start_frame,
                self.end_frame,
                int(self.config.always_reset_to_first_frame),
                self.first_frame_probability,
                self.immediate_first_frame_probability,
                self.config.reset_freeze_steps,
                curriculum_voc_scale,
                self.config.reset_voc_scale,
                self.config.shoulder_spread,
                self.left_shoulder_q_id,
                self.right_shoulder_q_id,
                self.finger_q_ids,
                self.config.seed,
            ],
            outputs=[
                self.reset_count,
                self.reset_frame,
                self.finger_scale,
                self.steps_since_reset,
                self.applied_voc_scale,
                self.raw_joint_q_offset,
                self.reference_joint_q_offset,
            ],
        )

    def prepare_explicit(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        self._require_bound()
        wp.launch(
            prepare_explicit_recon_body_reset,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.q_per_world,
                self.config.reset_freeze_steps + 1,
                curriculum_voc_scale,
            ],
            outputs=[
                self.finger_scale,
                self.steps_since_reset,
                self.applied_voc_scale,
                self.raw_joint_q_offset,
                self.reference_joint_q_offset,
            ],
        )

    def advance(
        self,
        terminated: wp.array,
        truncated: wp.array,
        timestep: wp.array,
        episode_step: wp.array,
        curriculum_voc_scale: wp.array,
    ) -> None:
        self._require_bound()
        wp.launch(
            advance_recon_body_reset,
            dim=self.world_count,
            inputs=[
                terminated,
                truncated,
                self.config.reset_freeze_steps,
                self.config.reset_freeze_steps - self.config.voc_decay_steps,
                self.config.voc_decay_steps,
                self.config.reset_voc_scale,
                self.q_per_world,
                curriculum_voc_scale,
                self.raw_joint_q_offset,
            ],
            outputs=[
                self.steps_since_reset,
                self.applied_voc_scale,
                self.reference_joint_q_offset,
                timestep,
                episode_step,
            ],
        )

    def sync_curriculum(self, curriculum_voc_scale: wp.array) -> None:
        wp.launch(
            sync_recon_body_voc,
            dim=self.world_count,
            inputs=[self.config.reset_freeze_steps, curriculum_voc_scale, self.steps_since_reset],
            outputs=[self.applied_voc_scale],
        )

    def compute_context(self, curriculum_voc_scale: wp.array, output: wp.array) -> wp.array:
        expected = (self.world_count * len(self.context_names),)
        if output.shape != expected or output.dtype != wp.float32:
            raise ValueError(f"ReconBody reset context must be float32 with shape {expected}")
        wp.launch(
            publish_recon_body_reset_context,
            dim=self.world_count,
            inputs=[
                self.applied_voc_scale,
                curriculum_voc_scale,
                self.steps_since_reset,
                self.config.reset_freeze_steps,
            ],
            outputs=[output],
        )
        return output
