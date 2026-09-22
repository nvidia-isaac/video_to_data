# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-tracking termination strategies and their shared runtime protocols."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import warp as wp

from flash_chord.embodiments.binding import DeviceRobotReference
from flash_chord.embodiments.frames import DeviceBodyFrameMap
from flash_chord.scene.builder import Scene
from flash_chord.utils.quat import quat_geodesic_angle, quat_mul_xyzw


@dataclass(frozen=True)
class ThresholdTerminationTermConfig:
    """Enablement and finite non-negative threshold for one tracking-failure condition."""

    threshold: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError(f"termination threshold must be finite and non-negative, got {self.threshold}")


@dataclass(frozen=True)
class ReferenceEndTerminationTermConfig:
    """Whether exhausting the reference trajectory truncates the episode."""

    enabled: bool = True


@dataclass(frozen=True)
class TrackingTerminationConfig:
    """Named reference-tracking failures and reference-end truncation."""

    wrist_position: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(threshold=0.15)
    wrist_orientation: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(
        threshold=0.0,
        enabled=False,
    )
    object_position: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(threshold=0.10)
    object_orientation: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(threshold=0.50)
    reference_end: ReferenceEndTerminationTermConfig = ReferenceEndTerminationTermConfig()

    def build(
        self,
        scene: Scene,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        num_objects: int,
        device=None,
        episode_step: wp.array | None = None,
    ) -> Termination:
        """Build the fused reference-tracking termination strategy."""
        return TrackingTermination.build(
            scene,
            config=self,
            object_body_ids_w=object_body_ids_w,
            object_ref_pos_w=object_ref_pos_w,
            object_ref_quat_w=object_ref_quat_w,
            episode_step=episode_step,
            num_objects=num_objects,
            device=device,
        )


@runtime_checkable
class Termination(Protocol):
    """Device termination strategy consumed by :class:`BaseEnv`."""

    terminated: wp.array
    truncated: wp.array

    def evaluate(self, body_q: wp.array, timestep: wp.array) -> None:
        """Write completion flags for the current transition."""
        ...


@runtime_checkable
class TerminationSpec(Protocol):
    """Setup-time termination builder selected by configuration."""

    def build(
        self,
        scene: Scene,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        num_objects: int,
        device=None,
        episode_step: wp.array | None = None,
    ) -> Termination:
        """Bind runtime dependencies and return a termination strategy."""
        ...


@runtime_checkable
class TerminationDiagnostics(Protocol):
    """Optional named cause masks and tracking errors for rollout diagnostics."""

    cause_names: tuple[str, ...]
    cause_arrays: tuple[wp.array, ...]
    error_names: tuple[str, ...]
    error_arrays: tuple[wp.array, ...]


@runtime_checkable
class PackedTerminationDiagnostics(Protocol):
    """Optional row-major float diagnostics exposed through one device array.

    Each world occupies one row whose columns follow ``packed_diagnostic_names``.
    """

    packed_diagnostic_names: tuple[str, ...]
    packed_diagnostics: wp.array


_TRACKING_CAUSE_NAMES = ("wrist", "object")
_TRACKING_ERROR_NAMES = (
    "wrist_position_m",
    "wrist_orientation_rad",
    "object_position_m",
    "object_orientation_rad",
)
_TRACKING_DIAGNOSTIC_WIDTH = len(_TRACKING_CAUSE_NAMES) + len(_TRACKING_ERROR_NAMES)


@wp.kernel
def termination_kernel(
    body_q: wp.array(dtype=wp.transform),
    timestep: wp.array(dtype=wp.int32),
    reference_num_frames: int,
    bodies_per_world: int,
    num_hands: int,
    wrist_frame_body_id: wp.array(dtype=wp.int32),  # [num_hands] retained per-world body id
    wrist_body_to_frame_pos: wp.array(dtype=wp.vec3),  # [num_hands] semantic wrist offset
    wrist_body_to_frame_quat: wp.array(dtype=wp.quat),  # [num_hands] semantic wrist rotation
    ref_wrist_pos: wp.array(dtype=wp.vec3),  # [T*num_hands] frame-major
    ref_wrist_quat: wp.array(dtype=wp.quat),  # [T*num_hands] frame-major, xyzw
    num_objects: int,
    object_body_id: wp.array(dtype=wp.int32),  # [W*num_objects] absolute body id, w-major
    ref_object_pos: wp.array(dtype=wp.vec3),  # [W*num_objects]
    ref_object_quat: wp.array(dtype=wp.quat),  # [W*num_objects] xyzw
    enable_wrist_pos: int,
    wrist_pos_thresh: float,
    enable_wrist_ori: int,
    wrist_ori_thresh: float,
    enable_object_pos: int,
    object_pos_thresh: float,
    enable_object_ori: int,
    object_ori_thresh: float,
    diagnostic_width: int,
    wrist_pos_err: wp.array(dtype=wp.float32),  # [W*num_hands] out
    wrist_ori_err: wp.array(dtype=wp.float32),  # out
    object_pos_err: wp.array(dtype=wp.float32),  # [W*num_objects] out
    object_ori_err: wp.array(dtype=wp.float32),  # out
    wrist_done: wp.array(dtype=wp.int32),  # [W] out
    object_done: wp.array(dtype=wp.int32),  # [W] out
    packed_diagnostics: wp.array(dtype=wp.float32),  # [W*6] row-major out: causes, then max errors
) -> None:
    """One thread per world: pose error of each wrist/object vs the reference, and the per-world flags."""
    w = wp.tid()
    base = w * bodies_per_world
    frame = timestep[w]
    if frame < 0:
        frame = 0
    if frame >= reference_num_frames:
        frame = reference_num_frames - 1

    wd = int(0)
    max_wrist_pos_err = float(0.0)
    max_wrist_ori_err = float(0.0)
    for h in range(num_hands):
        i = w * num_hands + h
        reference_i = frame * num_hands + h
        body_xf = body_q[base + wrist_frame_body_id[h]]
        body_quat = wp.transform_get_rotation(body_xf)
        frame_pos = wp.transform_point(body_xf, wrist_body_to_frame_pos[h])
        frame_quat = wp.normalize(quat_mul_xyzw(body_quat, wrist_body_to_frame_quat[h]))
        dp = wp.length(frame_pos - ref_wrist_pos[reference_i])
        do = quat_geodesic_angle(frame_quat, ref_wrist_quat[reference_i])
        wrist_pos_err[i] = dp
        wrist_ori_err[i] = do
        max_wrist_pos_err = wp.max(max_wrist_pos_err, dp)
        max_wrist_ori_err = wp.max(max_wrist_ori_err, do)
        if (enable_wrist_pos != 0 and dp > wrist_pos_thresh) or (enable_wrist_ori != 0 and do > wrist_ori_thresh):
            wd = 1
    wrist_done[w] = wd

    od = int(0)
    max_object_pos_err = float(0.0)
    max_object_ori_err = float(0.0)
    for o in range(num_objects):
        i = w * num_objects + o
        xf = body_q[object_body_id[i]]  # absolute id already carries the world offset
        dp = wp.length(wp.transform_get_translation(xf) - ref_object_pos[i])
        do = quat_geodesic_angle(wp.transform_get_rotation(xf), ref_object_quat[i])
        object_pos_err[i] = dp
        object_ori_err[i] = do
        max_object_pos_err = wp.max(max_object_pos_err, dp)
        max_object_ori_err = wp.max(max_object_ori_err, do)
        if (enable_object_pos != 0 and dp > object_pos_thresh) or (enable_object_ori != 0 and do > object_ori_thresh):
            od = 1
    object_done[w] = od

    diagnostic_base = w * diagnostic_width
    packed_diagnostics[diagnostic_base] = float(wd)
    packed_diagnostics[diagnostic_base + 1] = float(od)
    packed_diagnostics[diagnostic_base + 2] = max_wrist_pos_err
    packed_diagnostics[diagnostic_base + 3] = max_wrist_ori_err
    packed_diagnostics[diagnostic_base + 4] = max_object_pos_err
    packed_diagnostics[diagnostic_base + 5] = max_object_ori_err


@dataclass
class TerminationBuffers:
    """Per-world device outputs and semantic wrist-frame maps, sized to the scene.

    Each wrist frame is represented by one retained body plus a constant body-local
    transform. The kernel therefore remains independent of whether fixed semantic
    links were retained or collapsed during embodiment construction.
    """

    world_count: int
    num_hands: int
    num_objects: int
    bodies_per_world: int
    wrist_frames: DeviceBodyFrameMap
    wrist_pos_err: wp.array  # float32 [W*num_hands]
    wrist_ori_err: wp.array
    object_pos_err: wp.array  # float32 [W*num_objects]
    object_ori_err: wp.array
    wrist_done: wp.array  # int32 [W]
    object_done: wp.array  # int32 [W]
    packed_diagnostics: wp.array  # float32 [W*6], row-major causes followed by max errors

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        sides: tuple[str, ...] | None = None,
        num_objects: int | None = None,
        device=None,
    ) -> "TerminationBuffers":
        hands = scene.layout.hands if sides is None else tuple(scene.layout.hand(side) for side in sides)
        wrist_frames = [hand.palm_frame for hand in hands]
        n_hands = len(wrist_frames)
        n_obj = len(scene.objects) if num_objects is None else num_objects
        n_worlds = scene.world_count
        bodies_per_world = scene.model.body_count // n_worlds
        f32 = lambda n: wp.zeros(n, dtype=wp.float32, device=device)  # noqa: E731
        return cls(
            world_count=n_worlds,
            num_hands=n_hands,
            num_objects=n_obj,
            bodies_per_world=bodies_per_world,
            wrist_frames=DeviceBodyFrameMap.build(wrist_frames, device=device),
            wrist_pos_err=f32(n_worlds * n_hands),
            wrist_ori_err=f32(n_worlds * n_hands),
            object_pos_err=f32(n_worlds * n_obj),
            object_ori_err=f32(n_worlds * n_obj),
            wrist_done=wp.zeros(n_worlds, dtype=wp.int32, device=device),
            object_done=wp.zeros(n_worlds, dtype=wp.int32, device=device),
            packed_diagnostics=f32(n_worlds * _TRACKING_DIAGNOSTIC_WIDTH),
        )


def evaluate_termination(
    buf: TerminationBuffers,
    body_q: wp.array,
    timestep: wp.array,
    reference: DeviceRobotReference,
    object_body_ids: wp.array,
    ref_object_pos: wp.array,
    ref_object_quat: wp.array,
    config: TrackingTerminationConfig | None = None,
) -> None:
    """Launch the termination kernel into ``buf`` (per-world errors + ``done`` masks on device)."""
    config = config or TrackingTerminationConfig()
    wp.launch(
        termination_kernel,
        dim=buf.world_count,
        inputs=[
            body_q,
            timestep,
            reference.num_frames,
            buf.bodies_per_world,
            buf.num_hands,
            buf.wrist_frames.body_ids,
            buf.wrist_frames.body_to_frame_pos,
            buf.wrist_frames.body_to_frame_quat,
            reference.wrist_pos_w,
            reference.wrist_quat_w,
            buf.num_objects,
            object_body_ids,
            ref_object_pos,
            ref_object_quat,
            int(config.wrist_position.enabled),
            config.wrist_position.threshold,
            int(config.wrist_orientation.enabled),
            config.wrist_orientation.threshold,
            int(config.object_position.enabled),
            config.object_position.threshold,
            int(config.object_orientation.enabled),
            config.object_orientation.threshold,
            _TRACKING_DIAGNOSTIC_WIDTH,
        ],
        outputs=[
            buf.wrist_pos_err,
            buf.wrist_ori_err,
            buf.object_pos_err,
            buf.object_ori_err,
            buf.wrist_done,
            buf.object_done,
            buf.packed_diagnostics,
        ],
    )


@wp.kernel
def publish_tracking_completion(
    wrist_done: wp.array(dtype=wp.int32),
    object_done: wp.array(dtype=wp.int32),
    timestep: wp.array(dtype=wp.int32),
    last_frame: int,
    truncate_at_reference_end: int,
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
) -> None:
    """Aggregate tracking failures and reference exhaustion per world."""
    world = wp.tid()
    terminated[world] = wp.max(wrist_done[world], object_done[world])
    truncated[world] = int(0)
    if truncate_at_reference_end != 0 and timestep[world] >= last_frame:
        truncated[world] = 1


@dataclass
class TrackingTermination:
    """Reference wrist/object tracking termination with bound device targets."""

    buffers: TerminationBuffers
    config: TrackingTerminationConfig
    last_frame: int
    reference: DeviceRobotReference
    object_body_ids_w: wp.array
    object_ref_pos_w: wp.array
    object_ref_quat_w: wp.array
    terminated: wp.array
    truncated: wp.array

    @property
    def cause_names(self) -> tuple[str, ...]:
        return _TRACKING_CAUSE_NAMES

    @property
    def cause_arrays(self) -> tuple[wp.array, ...]:
        return (self.buffers.wrist_done, self.buffers.object_done)

    @property
    def error_names(self) -> tuple[str, ...]:
        return _TRACKING_ERROR_NAMES

    @property
    def error_arrays(self) -> tuple[wp.array, ...]:
        return (
            self.buffers.wrist_pos_err,
            self.buffers.wrist_ori_err,
            self.buffers.object_pos_err,
            self.buffers.object_ori_err,
        )

    @property
    def packed_diagnostic_names(self) -> tuple[str, ...]:
        return self.cause_names + self.error_names

    @property
    def packed_diagnostics(self) -> wp.array:
        return self.buffers.packed_diagnostics

    @classmethod
    def build(
        cls,
        scene: Scene,
        config: TrackingTerminationConfig,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        num_objects: int,
        device=None,
        episode_step: wp.array | None = None,
    ) -> "TrackingTermination":
        """Bind tracking inputs and allocate completion outputs."""
        return cls(
            buffers=TerminationBuffers.from_scene(
                scene,
                sides=scene.layout.sides,
                num_objects=num_objects,
                device=device,
            ),
            config=config,
            last_frame=scene.robot_reference.num_frames - 1,
            reference=DeviceRobotReference.build(scene.robot_reference, scene.layout, device=device),
            object_body_ids_w=object_body_ids_w,
            object_ref_pos_w=object_ref_pos_w,
            object_ref_quat_w=object_ref_quat_w,
            terminated=wp.zeros(scene.world_count, dtype=wp.int32, device=device),
            truncated=wp.zeros(scene.world_count, dtype=wp.int32, device=device),
        )

    def evaluate(self, body_q: wp.array, timestep: wp.array) -> None:
        evaluate_termination(
            self.buffers,
            body_q,
            timestep,
            self.reference,
            self.object_body_ids_w,
            self.object_ref_pos_w,
            self.object_ref_quat_w,
            self.config,
        )
        wp.launch(
            publish_tracking_completion,
            dim=self.buffers.world_count,
            inputs=[
                self.buffers.wrist_done,
                self.buffers.object_done,
                timestep,
                self.last_frame,
                int(self.config.reference_end.enabled),
            ],
            outputs=[self.terminated, self.truncated],
        )


@dataclass
class TerminationResult:
    """Host readback of :class:`TerminationBuffers` for the replay verification (peaks + any-done)."""

    max_wrist_pos: float
    max_wrist_ori: float
    max_object_pos: float
    max_object_ori: float
    wrist_done: bool
    object_done: bool

    @property
    def any(self) -> bool:
        return self.wrist_done or self.object_done


def read_termination(buf: TerminationBuffers) -> TerminationResult:
    """Read the device buffers back to the host (verification only — a sync; not for the runtime step)."""
    wpos, wori = buf.wrist_pos_err.numpy(), buf.wrist_ori_err.numpy()
    opos, oori = buf.object_pos_err.numpy(), buf.object_ori_err.numpy()
    peak = lambda a: float(a.max()) if a.size else 0.0  # noqa: E731
    return TerminationResult(
        max_wrist_pos=peak(wpos),
        max_wrist_ori=peak(wori),
        max_object_pos=peak(opos),
        max_object_ori=peak(oori),
        wrist_done=bool(buf.wrist_done.numpy().any()),
        object_done=bool(buf.object_done.numpy().any()),
    )


# ReconBody termination strategy.


RECON_BODY_TERMINATION_CAUSE_NAMES = ("pelvis", "palm", "object")
RECON_BODY_TERMINATION_ERROR_NAMES = (
    "pelvis_position_m",
    "pelvis_orientation_rad",
    "palm_position_m",
    "palm_orientation_rad",
    "object_position_m",
    "object_orientation_rad",
)


@dataclass(frozen=True)
class ReconBodyTerminationConfig:
    """BodyRecon failure thresholds and reset-freeze masking."""

    pelvis_position_threshold: float = 0.70
    pelvis_orientation_threshold: float = 1.50
    palm_position_threshold: float = 0.15
    palm_orientation_threshold: float = 1.50
    object_position_threshold: float = 0.10
    object_orientation_threshold: float = 1.50
    reset_freeze_steps: int = 50
    truncate_at_reference_end: bool = True
    sides: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        thresholds = (
            self.pelvis_position_threshold,
            self.pelvis_orientation_threshold,
            self.palm_position_threshold,
            self.palm_orientation_threshold,
            self.object_position_threshold,
            self.object_orientation_threshold,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in thresholds):
            raise ValueError("ReconBody termination thresholds must be finite and non-negative")
        if self.reset_freeze_steps < 0:
            raise ValueError("ReconBody termination reset freeze must be non-negative")
        object.__setattr__(self, "sides", tuple(self.sides))

    def build(
        self,
        scene: Scene,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        num_objects: int,
        device=None,
        episode_step: wp.array | None = None,
    ) -> ReconBodyTermination:
        if episode_step is None:
            raise ValueError("ReconBody termination requires the per-world episode step")
        return ReconBodyTermination.build(
            scene,
            object_body_ids_w,
            object_ref_pos_w,
            object_ref_quat_w,
            episode_step,
            num_objects,
            config=self,
            device=device,
        )


@wp.kernel
def evaluate_recon_body_termination(
    body_q: wp.array(dtype=wp.transform),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    reference_joint_q: wp.array(dtype=wp.float32, ndim=2),
    reference_wrist_pos_w: wp.array(dtype=wp.vec3),
    reference_wrist_quat_w: wp.array(dtype=wp.quat),
    wrist_body_ids: wp.array(dtype=wp.int32),
    wrist_local_pos: wp.array(dtype=wp.vec3),
    wrist_local_quat: wp.array(dtype=wp.quat),
    object_body_ids_w: wp.array(dtype=wp.int32),
    object_ref_pos_w: wp.array(dtype=wp.vec3),
    object_ref_quat_w: wp.array(dtype=wp.quat),
    pelvis_body_id: int,
    num_frames: int,
    bodies_per_world: int,
    num_hands: int,
    num_objects: int,
    reset_freeze_steps: int,
    pelvis_position_threshold: float,
    pelvis_orientation_threshold: float,
    palm_position_threshold: float,
    palm_orientation_threshold: float,
    object_position_threshold: float,
    object_orientation_threshold: float,
    truncate_at_reference_end: int,
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    pelvis_done: wp.array(dtype=wp.int32),
    palm_done: wp.array(dtype=wp.int32),
    object_done: wp.array(dtype=wp.int32),
    pelvis_position_error_out: wp.array(dtype=wp.float32),
    pelvis_orientation_error_out: wp.array(dtype=wp.float32),
    palm_position_error_out: wp.array(dtype=wp.float32),
    palm_orientation_error_out: wp.array(dtype=wp.float32),
    object_position_error_out: wp.array(dtype=wp.float32),
    object_orientation_error_out: wp.array(dtype=wp.float32),
    packed_diagnostics: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    body_base = world * bodies_per_world
    pelvis_xf = body_q[body_base + pelvis_body_id]
    pelvis_position = wp.transform_get_translation(pelvis_xf)
    pelvis_orientation = wp.normalize(wp.transform_get_rotation(pelvis_xf))
    reference_pelvis_position = wp.vec3(
        reference_joint_q[frame, 0],
        reference_joint_q[frame, 1],
        reference_joint_q[frame, 2],
    )
    reference_pelvis_orientation = wp.normalize(
        wp.quat(
            reference_joint_q[frame, 3],
            reference_joint_q[frame, 4],
            reference_joint_q[frame, 5],
            reference_joint_q[frame, 6],
        )
    )
    pelvis_position_error = wp.length(pelvis_position - reference_pelvis_position)
    pelvis_orientation_error = quat_geodesic_angle(pelvis_orientation, reference_pelvis_orientation)
    pelvis_failed = int(
        pelvis_position_error > pelvis_position_threshold or pelvis_orientation_error > pelvis_orientation_threshold
    )

    max_palm_position_error = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    max_palm_orientation_error = float(0.0)  # noqa: UP018
    for hand in range(num_hands):
        body_xf = body_q[body_base + wrist_body_ids[hand]]
        palm_position = wp.transform_point(body_xf, wrist_local_pos[hand])
        palm_orientation = wp.normalize(quat_mul_xyzw(wp.transform_get_rotation(body_xf), wrist_local_quat[hand]))
        max_palm_position_error = wp.max(
            max_palm_position_error,
            wp.length(palm_position - reference_wrist_pos_w[frame * num_hands + hand]),
        )
        max_palm_orientation_error = wp.max(
            max_palm_orientation_error,
            quat_geodesic_angle(palm_orientation, reference_wrist_quat_w[frame * num_hands + hand]),
        )

    max_object_position_error = float(0.0)  # noqa: UP018
    max_object_orientation_error = float(0.0)  # noqa: UP018
    for object_id in range(num_objects):
        index = world * num_objects + object_id
        object_xf = body_q[object_body_ids_w[index]]
        max_object_position_error = wp.max(
            max_object_position_error,
            wp.length(wp.transform_get_translation(object_xf) - object_ref_pos_w[index]),
        )
        max_object_orientation_error = wp.max(
            max_object_orientation_error,
            quat_geodesic_angle(wp.transform_get_rotation(object_xf), object_ref_quat_w[index]),
        )

    freeze_masked = episode_step[world] <= reset_freeze_steps
    palm_failed = int(0)  # noqa: UP018,RUF046 - Warp requires explicitly typed mutable locals.
    object_failed = int(0)  # noqa: UP018,RUF046
    if not freeze_masked:
        palm_failed = int(
            max_palm_position_error > palm_position_threshold or max_palm_orientation_error > palm_orientation_threshold
        )
        object_failed = int(
            max_object_position_error > object_position_threshold
            or max_object_orientation_error > object_orientation_threshold
        )

    pelvis_done[world] = pelvis_failed
    palm_done[world] = palm_failed
    object_done[world] = object_failed
    terminated[world] = wp.max(pelvis_failed, wp.max(palm_failed, object_failed))
    truncated[world] = int(truncate_at_reference_end != 0 and timestep[world] >= num_frames - 1)

    pelvis_position_error_out[world] = pelvis_position_error
    pelvis_orientation_error_out[world] = pelvis_orientation_error
    palm_position_error_out[world] = max_palm_position_error
    palm_orientation_error_out[world] = max_palm_orientation_error
    object_position_error_out[world] = max_object_position_error
    object_orientation_error_out[world] = max_object_orientation_error
    base = world * 9
    packed_diagnostics[base] = float(pelvis_failed)
    packed_diagnostics[base + 1] = float(palm_failed)
    packed_diagnostics[base + 2] = float(object_failed)
    packed_diagnostics[base + 3] = pelvis_position_error
    packed_diagnostics[base + 4] = pelvis_orientation_error
    packed_diagnostics[base + 5] = max_palm_position_error
    packed_diagnostics[base + 6] = max_palm_orientation_error
    packed_diagnostics[base + 7] = max_object_position_error
    packed_diagnostics[base + 8] = max_object_orientation_error


@dataclass
class ReconBodyTermination:
    """Fused BodyRecon failure evaluation with reset-age masking."""

    config: ReconBodyTerminationConfig
    world_count: int
    bodies_per_world: int
    pelvis_body_id: int
    num_objects: int
    episode_step: wp.array
    reference: DeviceRobotReference
    wrist_frames: DeviceBodyFrameMap
    object_body_ids_w: wp.array
    object_ref_pos_w: wp.array
    object_ref_quat_w: wp.array
    terminated: wp.array
    truncated: wp.array
    pelvis_done: wp.array
    palm_done: wp.array
    object_done: wp.array
    pelvis_position_error: wp.array
    pelvis_orientation_error: wp.array
    palm_position_error: wp.array
    palm_orientation_error: wp.array
    object_position_error: wp.array
    object_orientation_error: wp.array
    packed_diagnostics: wp.array

    @classmethod
    def build(
        cls,
        scene: Scene,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        episode_step: wp.array,
        num_objects: int,
        config: ReconBodyTerminationConfig | None = None,
        device=None,
    ) -> ReconBodyTermination:
        config = config or ReconBodyTerminationConfig()
        if num_objects != 1:
            raise ValueError("ReconBody termination currently requires exactly one rigid object")
        pelvis = tuple(frame for frame in scene.layout.semantic_frames if frame.name == "pelvis")
        if len(pelvis) != 1:
            raise ValueError("ReconBody termination requires one pelvis semantic frame")
        reference = DeviceRobotReference.build(
            scene.robot_reference,
            scene.layout,
            sides=config.sides,
            device=device,
        )
        wrist_frames = DeviceBodyFrameMap.build(
            [scene.layout.hand(side).palm_frame for side in config.sides],
            device=device,
        )

        def i32() -> wp.array:
            return wp.zeros(scene.world_count, dtype=wp.int32, device=device)

        def f32() -> wp.array:
            return wp.zeros(scene.world_count, dtype=wp.float32, device=device)

        return cls(
            config=config,
            world_count=scene.world_count,
            bodies_per_world=scene.model.body_count // scene.world_count,
            pelvis_body_id=pelvis[0].body_id,
            num_objects=num_objects,
            episode_step=episode_step,
            reference=reference,
            wrist_frames=wrist_frames,
            object_body_ids_w=object_body_ids_w,
            object_ref_pos_w=object_ref_pos_w,
            object_ref_quat_w=object_ref_quat_w,
            terminated=i32(),
            truncated=i32(),
            pelvis_done=i32(),
            palm_done=i32(),
            object_done=i32(),
            pelvis_position_error=f32(),
            pelvis_orientation_error=f32(),
            palm_position_error=f32(),
            palm_orientation_error=f32(),
            object_position_error=f32(),
            object_orientation_error=f32(),
            packed_diagnostics=wp.zeros(scene.world_count * 9, dtype=wp.float32, device=device),
        )

    @property
    def cause_names(self) -> tuple[str, ...]:
        return RECON_BODY_TERMINATION_CAUSE_NAMES

    @property
    def cause_arrays(self) -> tuple[wp.array, ...]:
        return (self.pelvis_done, self.palm_done, self.object_done)

    @property
    def error_names(self) -> tuple[str, ...]:
        return RECON_BODY_TERMINATION_ERROR_NAMES

    @property
    def error_arrays(self) -> tuple[wp.array, ...]:
        return (
            self.pelvis_position_error,
            self.pelvis_orientation_error,
            self.palm_position_error,
            self.palm_orientation_error,
            self.object_position_error,
            self.object_orientation_error,
        )

    @property
    def packed_diagnostic_names(self) -> tuple[str, ...]:
        return self.cause_names + self.error_names

    def evaluate(self, body_q: wp.array, timestep: wp.array) -> None:
        wp.launch(
            evaluate_recon_body_termination,
            dim=self.world_count,
            inputs=[
                body_q,
                timestep,
                self.episode_step,
                self.reference.joint_q,
                self.reference.wrist_pos_w,
                self.reference.wrist_quat_w,
                self.wrist_frames.body_ids,
                self.wrist_frames.body_to_frame_pos,
                self.wrist_frames.body_to_frame_quat,
                self.object_body_ids_w,
                self.object_ref_pos_w,
                self.object_ref_quat_w,
                self.pelvis_body_id,
                self.reference.num_frames,
                self.bodies_per_world,
                2,
                self.num_objects,
                self.config.reset_freeze_steps,
                self.config.pelvis_position_threshold,
                self.config.pelvis_orientation_threshold,
                self.config.palm_position_threshold,
                self.config.palm_orientation_threshold,
                self.config.object_position_threshold,
                self.config.object_orientation_threshold,
                int(self.config.truncate_at_reference_end),
            ],
            outputs=[
                self.terminated,
                self.truncated,
                self.pelvis_done,
                self.palm_done,
                self.object_done,
                self.pelvis_position_error,
                self.pelvis_orientation_error,
                self.palm_position_error,
                self.palm_orientation_error,
                self.object_position_error,
                self.object_orientation_error,
                self.packed_diagnostics,
            ],
        )
