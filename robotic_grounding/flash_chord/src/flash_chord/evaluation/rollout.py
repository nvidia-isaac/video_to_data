# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched object-pose recording for one deterministic evaluation cohort.

Per-step state accumulates into preallocated device buffers and is read back once at the end.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.evaluation.metrics import ADD_VERTEX_COUNT, ADD_VERTEX_SEED
from flash_chord.runtime.command import ObjectLayout
from flash_chord.utils.quat import quat_rotate_xyzw, wxyz_to_xyzw

_POSE_WIDTH = 7


@dataclass(frozen=True, slots=True)
class CompletionOutcome:
    """Per-world episode outcome latched at each world's first completion."""

    completion_step: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    reference_progress: np.ndarray


@dataclass(frozen=True, slots=True)
class ObjectPoseRollout:
    """One recorded cohort: object poses, tracking errors, and completion outcome.

    Poses are position followed by an **xyzw** quaternion, ordered by reference body id.
    """

    achieved_pose_w: np.ndarray
    reference_pose_w: np.ndarray
    tracking_error: np.ndarray
    tracking_error_names: tuple[str, ...]
    object_vertices_o: np.ndarray
    body_object_ids: np.ndarray
    object_body_names: tuple[str, ...]
    completion: CompletionOutcome
    control_fps: float
    start_frame: int
    non_finite_worlds: tuple[int, ...]


@wp.kernel
def record_object_pose(
    body_q: wp.array(dtype=wp.transform),
    object_body_ids_w: wp.array(dtype=wp.int32),  # [W*B] absolute body ids, reference-body order
    step: int,
    slot_count: int,  # W*B
    pose_out: wp.array(dtype=wp.float32),  # [T*W*B*7] out, position then xyzw quaternion
) -> None:
    """Gather every object body's world transform into one step slice."""
    slot = wp.tid()
    transform = body_q[object_body_ids_w[slot]]
    position = wp.transform_get_translation(transform)
    rotation = wp.transform_get_rotation(transform)
    base = (step * slot_count + slot) * 7
    pose_out[base + 0] = position[0]
    pose_out[base + 1] = position[1]
    pose_out[base + 2] = position[2]
    pose_out[base + 3] = rotation[0]
    pose_out[base + 4] = rotation[1]
    pose_out[base + 5] = rotation[2]
    pose_out[base + 6] = rotation[3]


@wp.kernel
def record_tracking_error(
    packed_diagnostics: wp.array(dtype=wp.float32),  # [W*diagnostic_width], causes then max errors
    diagnostic_width: int,
    cause_count: int,
    error_count: int,
    step: int,
    world_count: int,
    error_out: wp.array(dtype=wp.float32),  # [T*W*error_count] out
) -> None:
    """Copy the per-world maximum tracking errors into one step slice."""
    world = wp.tid()
    source = world * diagnostic_width + cause_count
    base = (step * world_count + world) * error_count
    for term in range(error_count):
        error_out[base + term] = packed_diagnostics[source + term]


@wp.kernel
def latch_completion(
    done: wp.array(dtype=wp.int32),
    truncation: wp.array(dtype=wp.int32),
    reference_progress: wp.array(dtype=wp.float32),
    step: int,
    completion_step: wp.array(dtype=wp.int32),  # out, preset to -1
    completed_terminated: wp.array(dtype=wp.int32),
    completed_truncated: wp.array(dtype=wp.int32),
    completed_progress: wp.array(dtype=wp.float32),
) -> None:
    """Record each world's first completion without ending its rollout."""
    world = wp.tid()
    if completion_step[world] >= 0:
        return
    if done[world] != 0 or truncation[world] != 0:
        completion_step[world] = step
        completed_terminated[world] = done[world]
        completed_truncated[world] = truncation[world]
        completed_progress[world] = reference_progress[world]


def body_object_ids(layout: ObjectLayout) -> np.ndarray:
    """``(B,)`` owning object index per reference body, joined through the simulation body id."""
    reference_of_simulation = {body_id: reference for reference, body_id in enumerate(layout.body_ids)}
    ids = np.full(layout.num_bodies, -1, dtype=np.int64)
    offsets = layout.voc_object_body_offsets
    for object_index in range(layout.num_objects):
        for slot in range(offsets[object_index], offsets[object_index + 1]):
            ids[reference_of_simulation[layout.voc_object_body_ids[slot]]] = object_index
    if np.any(ids < 0):
        raise ValueError("object layout does not assign every reference body to an object")
    return ids


def reference_object_pose(reference: Reference, start_frame: int, step_count: int) -> np.ndarray:
    """``(T, B, 7)`` reference object poses, position then **xyzw** quaternion."""
    position = np.asarray(reference.object_body_pos_w(), dtype=np.float64)
    rotation = wxyz_to_xyzw(np.asarray(reference.object_body_quat_w(), dtype=np.float64))
    stop = start_frame + step_count
    if stop > position.shape[0]:
        raise ValueError(f"reference has {position.shape[0]} frames; rollout needs {stop}")
    return np.concatenate([position[start_frame:stop], rotation[start_frame:stop]], axis=-1)


def sample_object_vertices(
    scene,
    *,
    count: int = ADD_VERTEX_COUNT,
    seed: int = ADD_VERTEX_SEED,
) -> np.ndarray:
    """``(B, count, 3)`` body-frame vertices sampled from the shapes the simulation itself uses.

    Ordered by reference body id. Geometry is read from the finalized model with the scale and
    shape transform already applied, so it carries the same units and frame as ``body_q``.
    """
    import newton

    model = scene.model
    visible = int(newton.ShapeFlags.VISIBLE)
    mesh_type = int(newton.GeoType.MESH)
    flags = np.asarray(model.shape_flags.numpy())
    shape_body = np.asarray(model.shape_body.numpy())
    shape_type = np.asarray(model.shape_type.numpy())
    shape_scale = np.asarray(model.shape_scale.numpy(), dtype=np.float64)
    shape_transform = np.asarray(model.shape_transform.numpy(), dtype=np.float64)

    by_reference: dict[int, np.ndarray] = {}
    for binding in scene.objects:
        for body in binding.bodies:
            pieces = []
            for shape in range(binding.shapes.start, binding.shapes.stop):
                if shape_body[shape] != body.body_id or shape_type[shape] != mesh_type:
                    continue
                if not flags[shape] & visible:
                    continue
                source = model.shape_source[shape]
                if source is None or not hasattr(source, "vertices"):
                    continue
                scaled = np.asarray(source.vertices, dtype=np.float64) * shape_scale[shape][None, :]
                offset = shape_transform[shape]
                pieces.append(quat_rotate_xyzw(offset[3:7], scaled) + offset[:3][None, :])
            if not pieces:
                raise ValueError(f"object {binding.name!r} body {body.body_id} has no visible mesh shape")
            by_reference[body.reference_body_id] = np.concatenate(pieces)

    missing = set(range(len(by_reference))) - set(by_reference)
    if missing or not by_reference:
        raise ValueError(f"object bodies do not cover reference ids 0..{len(by_reference) - 1}")
    generator = np.random.default_rng(seed)
    sampled = np.empty((len(by_reference), count, 3), dtype=np.float64)
    for reference_id in sorted(by_reference):
        vertices = by_reference[reference_id]
        chosen = generator.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
        sampled[reference_id] = vertices[chosen]
    return sampled


def record_object_pose_rollout(
    env,
    jax_env,
    policy_action: Callable,
    actor_state,
    key,
    observation,
    *,
    reference: Reference,
    step_count: int,
    start_frame: int = 0,
    progress_interval: int = 50,
) -> ObjectPoseRollout:
    """Run one deterministic cohort and return its recorded poses, errors, and outcome.

    ``observation`` is the caller's post-reset observation. The tracking-failure termination
    terms must be disabled so every world keeps advancing through the whole reference.
    """
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    termination = env.termination
    cause_names = tuple(termination.cause_names)
    error_names = tuple(termination.error_names)
    diagnostic_names = tuple(termination.packed_diagnostic_names)
    if diagnostic_names != cause_names + error_names:
        raise ValueError(
            f"termination publishes {diagnostic_names}; expected causes {cause_names} followed by errors {error_names}"
        )
    error_count = len(error_names)
    if not error_count:
        raise ValueError("evaluation requires at least one tracking-error diagnostic")

    world_count = env.world_count
    body_ids_w = env.command.body_ids_w
    layout = env.command.layout
    bodies = layout.num_bodies
    slot_count = world_count * bodies
    device = env.device

    pose = wp.zeros(step_count * slot_count * _POSE_WIDTH, dtype=wp.float32, device=device)
    error = wp.zeros(step_count * world_count * error_count, dtype=wp.float32, device=device)
    completion_step = wp.full(world_count, -1, dtype=wp.int32, device=device)
    completed_terminated = wp.zeros(world_count, dtype=wp.int32, device=device)
    completed_truncated = wp.zeros(world_count, dtype=wp.int32, device=device)
    completed_progress = wp.zeros(world_count, dtype=wp.float32, device=device)

    for step in range(step_count):
        action, key = policy_action(actor_state, observation, key)
        observation = jax_env.step(action).observation
        # Explicit device: a launch off the arrays' device is silently skipped, not raised.
        wp.launch(
            record_object_pose,
            dim=slot_count,
            device=device,
            inputs=[env.state_0.body_q, body_ids_w, step, slot_count],
            outputs=[pose],
        )
        wp.launch(
            record_tracking_error,
            dim=world_count,
            device=device,
            inputs=[
                termination.packed_diagnostics,
                len(diagnostic_names),
                len(cause_names),
                error_count,
                step,
                world_count,
            ],
            outputs=[error],
        )
        wp.launch(
            latch_completion,
            dim=world_count,
            device=device,
            inputs=[env.done, env.truncation, env.episode_reference_progress, step],
            outputs=[completion_step, completed_terminated, completed_truncated, completed_progress],
        )
        if progress_interval and (step % progress_interval == 0 or step + 1 == step_count):
            print(f"evaluation step {step + 1}/{step_count}", flush=True)

    achieved = pose.numpy().reshape(step_count, world_count, bodies, _POSE_WIDTH).astype(np.float64)
    finite = np.isfinite(achieved).all(axis=(0, 2, 3))
    non_finite = tuple(int(world) for world in np.flatnonzero(~finite))
    if non_finite:
        achieved[:, ~finite] = 0.0
        print(f"warning: {len(non_finite)} worlds recorded non-finite poses and are excluded", flush=True)
    completion = CompletionOutcome(
        completion_step=completion_step.numpy().astype(np.int32),
        terminated=completed_terminated.numpy().astype(np.bool_),
        truncated=completed_truncated.numpy().astype(np.bool_),
        reference_progress=completed_progress.numpy().astype(np.float64),
    )
    return ObjectPoseRollout(
        achieved_pose_w=achieved,
        reference_pose_w=reference_object_pose(reference, start_frame, step_count),
        tracking_error=error.numpy().reshape(step_count, world_count, error_count).astype(np.float64),
        tracking_error_names=error_names,
        object_vertices_o=sample_object_vertices(env.scene),
        body_object_ids=body_object_ids(layout),
        object_body_names=tuple(reference.object_body_names()),
        completion=completion,
        control_fps=float(env.config.sim.fps),
        start_frame=start_frame,
        non_finite_worlds=non_finite,
    )
