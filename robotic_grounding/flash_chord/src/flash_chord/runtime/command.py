# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-side object command gathering from the reference trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.scene.builder import Scene
from flash_chord.scene.objects import ObjectBinding, ObjectJointBinding
from flash_chord.utils.quat import wxyz_to_xyzw


@dataclass(frozen=True)
class ObjectLayout:
    """Reference-body, simulated-body, VOC-root, and articulation mappings."""

    num_bodies: int
    num_objects: int
    num_articulations: int
    body_ids: tuple[int, ...]
    body_object_ids: tuple[int, ...]
    voc_body_ids: tuple[int, ...]
    voc_reference_body_ids: tuple[int, ...]
    voc_object_body_offsets: tuple[int, ...]
    voc_object_body_ids: tuple[int, ...]
    articulation_q_ids: tuple[int, ...]
    articulation_dof_ids: tuple[int, ...]
    articulation_reference_ids: tuple[int, ...]
    articulation_kp: tuple[float, ...]
    articulation_kd: tuple[float, ...]
    articulation_effort_limit: tuple[float, ...]

    @classmethod
    def from_bindings(
        cls,
        bindings: list[ObjectBinding],
        num_reference_bodies: int,
        num_reference_articulations: int,
    ) -> "ObjectLayout":
        body_ids: dict[int, int] = {}
        body_object_ids: dict[int, int] = {}
        simulation_body_ids: set[int] = set()
        voc_body_ids: list[int] = []
        voc_reference_body_ids: list[int] = []
        voc_object_body_offsets = [0]
        voc_object_body_ids: list[int] = []
        articulation_by_reference: dict[int, ObjectJointBinding] = {}
        articulation_q_ids_seen: set[int] = set()
        articulation_dof_ids_seen: set[int] = set()
        for object_id, binding in enumerate(bindings):
            for body in binding.bodies:
                if body.reference_body_id >= num_reference_bodies:
                    raise ValueError(
                        f"object {binding.name!r} maps reference body {body.reference_body_id}, "
                        f"but the reference has {num_reference_bodies} bodies"
                    )
                if body.reference_body_id in body_ids:
                    raise ValueError(f"reference object body {body.reference_body_id} is mapped more than once")
                if body.body_id in simulation_body_ids:
                    raise ValueError(f"simulated object body {body.body_id} is mapped more than once")
                body_ids[body.reference_body_id] = body.body_id
                body_object_ids[body.reference_body_id] = object_id
                simulation_body_ids.add(body.body_id)
            voc_body_ids.append(binding.root.body_id)
            voc_reference_body_ids.append(binding.root.reference_body_id)
            voc_object_body_ids.extend(body.body_id for body in binding.bodies)
            voc_object_body_offsets.append(len(voc_object_body_ids))
            for articulation in binding.articulations:
                if articulation.reference_id >= num_reference_articulations:
                    raise ValueError(
                        f"object {binding.name!r} maps articulation {articulation.reference_id}, "
                        f"but the reference has {num_reference_articulations} columns"
                    )
                if articulation.reference_id in articulation_by_reference:
                    raise ValueError(
                        f"reference object articulation {articulation.reference_id} is mapped more than once"
                    )
                if articulation.q_id in articulation_q_ids_seen or articulation.dof_id in articulation_dof_ids_seen:
                    raise ValueError(
                        "simulated object articulation q/DOF is mapped more than once: "
                        f"q={articulation.q_id}, dof={articulation.dof_id}"
                    )
                articulation_by_reference[articulation.reference_id] = articulation
                articulation_q_ids_seen.add(articulation.q_id)
                articulation_dof_ids_seen.add(articulation.dof_id)
        missing_bodies = tuple(
            reference_id for reference_id in range(num_reference_bodies) if reference_id not in body_ids
        )
        if missing_bodies:
            raise ValueError(f"unmapped reference object bodies: {missing_bodies}")
        missing_articulations = tuple(
            reference_id
            for reference_id in range(num_reference_articulations)
            if reference_id not in articulation_by_reference
        )
        if missing_articulations:
            raise ValueError(f"unmapped reference object articulations: {missing_articulations}")
        articulation_reference_ids = tuple(sorted(articulation_by_reference))
        articulations = tuple(articulation_by_reference[reference_id] for reference_id in articulation_reference_ids)
        return cls(
            num_bodies=num_reference_bodies,
            num_objects=len(bindings),
            num_articulations=len(articulations),
            body_ids=tuple(body_ids[reference_id] for reference_id in range(num_reference_bodies)),
            body_object_ids=tuple(body_object_ids[reference_id] for reference_id in range(num_reference_bodies)),
            voc_body_ids=tuple(voc_body_ids),
            voc_reference_body_ids=tuple(voc_reference_body_ids),
            voc_object_body_offsets=tuple(voc_object_body_offsets),
            voc_object_body_ids=tuple(voc_object_body_ids),
            articulation_q_ids=tuple(articulation.q_id for articulation in articulations),
            articulation_dof_ids=tuple(articulation.dof_id for articulation in articulations),
            articulation_reference_ids=articulation_reference_ids,
            articulation_kp=tuple(articulation.drive.kp for articulation in articulations),
            articulation_kd=tuple(articulation.drive.kd for articulation in articulations),
            articulation_effort_limit=tuple(articulation.drive.effort_limit for articulation in articulations),
        )


@dataclass
class ObjectReference:
    """Object pose and articulation trajectory on one Warp device."""

    num_frames: int
    num_bodies: int
    num_articulations: int
    body_pos_w: wp.array
    body_quat_w: wp.array
    articulation_joint_pos: wp.array

    @classmethod
    def build(cls, reference: Reference, device=None) -> "ObjectReference":
        body_pos_w = np.asarray(reference.object_body_pos_w(), dtype=np.float32)
        body_quat_w = np.asarray(wxyz_to_xyzw(reference.object_body_quat_w()), dtype=np.float32)
        articulation_joint_pos = np.asarray(reference.object_articulation(), dtype=np.float32)
        return cls(
            num_frames=reference.num_frames,
            num_bodies=body_pos_w.shape[1],
            num_articulations=articulation_joint_pos.shape[1],
            body_pos_w=wp.array(body_pos_w.reshape(-1, 3), dtype=wp.vec3, device=device),
            body_quat_w=wp.array(body_quat_w.reshape(-1, 4), dtype=wp.quat, device=device),
            articulation_joint_pos=wp.array(articulation_joint_pos.reshape(-1), dtype=wp.float32, device=device),
        )


@wp.kernel
def gather_object_command(
    timestep: wp.array(dtype=wp.int32),
    num_frames: int,
    num_bodies: int,
    num_objects: int,
    num_articulations: int,
    num_reference_articulations: int,
    ref_body_pos_w: wp.array(dtype=wp.vec3),
    ref_body_quat_w: wp.array(dtype=wp.quat),
    ref_articulation_joint_pos: wp.array(dtype=wp.float32),
    body_object_ids: wp.array(dtype=wp.int32),
    voc_reference_body_ids: wp.array(dtype=wp.int32),
    articulation_reference_ids: wp.array(dtype=wp.int32),
    object_scales: wp.array(dtype=wp.float32),
    object_root_position_offsets_w: wp.array(dtype=wp.vec3),
    body_target_pos_w: wp.array(dtype=wp.vec3),
    body_target_quat_w: wp.array(dtype=wp.quat),
    voc_target_pos_w: wp.array(dtype=wp.vec3),
    voc_target_quat_w: wp.array(dtype=wp.quat),
    articulation_target_pos: wp.array(dtype=wp.float32),
) -> None:
    """Gather one world's object targets at its current reference frame."""
    world = wp.tid()
    frame = timestep[world]
    if frame < 0:
        frame = 0
    if frame >= num_frames:
        frame = num_frames - 1

    reference_body_base = frame * num_bodies
    body_target_base = world * num_bodies
    for body in range(num_bodies):
        object_id = body_object_ids[body]
        root_reference_body = voc_reference_body_ids[object_id]
        root_position = ref_body_pos_w[reference_body_base + root_reference_body]
        scale = object_scales[world * num_objects + object_id]
        offset = object_root_position_offsets_w[world * num_objects + object_id]
        body_target_pos_w[body_target_base + body] = (
            root_position + offset + scale * (ref_body_pos_w[reference_body_base + body] - root_position)
        )
        body_target_quat_w[body_target_base + body] = ref_body_quat_w[reference_body_base + body]

    voc_target_base = world * num_objects
    for object_id in range(num_objects):
        reference_body = voc_reference_body_ids[object_id]
        voc_target_pos_w[voc_target_base + object_id] = (
            ref_body_pos_w[reference_body_base + reference_body]
            + object_root_position_offsets_w[world * num_objects + object_id]
        )
        voc_target_quat_w[voc_target_base + object_id] = ref_body_quat_w[reference_body_base + reference_body]

    if num_articulations > 0:
        reference_articulation_base = frame * num_reference_articulations
        articulation_target_base = world * num_articulations
        for articulation in range(num_articulations):
            articulation_target_pos[articulation_target_base + articulation] = ref_articulation_joint_pos[
                reference_articulation_base + articulation_reference_ids[articulation]
            ]


@dataclass
class CommandBuffers:
    """Object command mappings, reference, and per-world target buffers."""

    layout: ObjectLayout
    reference: ObjectReference
    world_count: int
    num_joint_q: int
    num_joint_dof: int
    bodies_per_world: int
    body_ids: wp.array
    body_object_ids: wp.array
    body_ids_w: wp.array
    voc_body_ids_w: wp.array
    voc_reference_body_ids: wp.array
    object_scales: wp.array
    object_root_position_offsets_w: wp.array
    object_radius: wp.array
    voc_object_body_offsets: wp.array
    voc_object_body_ids: wp.array
    articulation_q_ids: wp.array
    articulation_dof_ids: wp.array
    articulation_reference_ids: wp.array
    articulation_kp: wp.array
    articulation_kd: wp.array
    articulation_effort_limit: wp.array
    body_target_pos_w: wp.array
    body_target_quat_w: wp.array
    voc_target_pos_w: wp.array
    voc_target_quat_w: wp.array
    articulation_target_pos: wp.array

    @classmethod
    def build(cls, scene: Scene, reference: Reference, device=None) -> "CommandBuffers":
        object_reference = ObjectReference.build(reference, device=device)
        layout = ObjectLayout.from_bindings(
            scene.objects,
            object_reference.num_bodies,
            object_reference.num_articulations,
        )
        if layout.num_articulations != object_reference.num_articulations:
            raise ValueError(
                f"scene has {layout.num_articulations} articulated joints; "
                f"reference has {object_reference.num_articulations} columns"
            )
        bodies_per_world = scene.model.body_count // scene.world_count
        body_ids_w = [
            body_id + world * bodies_per_world for world in range(scene.world_count) for body_id in layout.body_ids
        ]
        voc_body_ids_w = [
            body_id + world * bodies_per_world for world in range(scene.world_count) for body_id in layout.voc_body_ids
        ]
        i32 = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
        f32 = lambda values: wp.array(values, dtype=wp.float32, device=device)  # noqa: E731
        object_scales = np.asarray(scene.object_scales, dtype=np.float32)
        root_offsets = np.asarray(
            scene.object_root_position_offsets_w,
            dtype=np.float32,
        )
        radius = np.asarray(reference.object_mesh_radius(), dtype=np.float32)
        if radius.shape != (layout.num_bodies,):
            raise ValueError(f"object_mesh_radius has shape {radius.shape}; " f"expected ({layout.num_bodies},)")
        body_scales = object_scales[
            :,
            np.asarray(layout.body_object_ids, dtype=np.int64),
        ]
        return cls(
            layout=layout,
            reference=object_reference,
            world_count=scene.world_count,
            num_joint_q=scene.model.joint_coord_count // scene.world_count,
            num_joint_dof=scene.model.joint_dof_count // scene.world_count,
            bodies_per_world=bodies_per_world,
            body_ids=i32(layout.body_ids),
            body_object_ids=i32(layout.body_object_ids),
            body_ids_w=i32(body_ids_w),
            voc_body_ids_w=i32(voc_body_ids_w),
            voc_reference_body_ids=i32(layout.voc_reference_body_ids),
            object_scales=f32(object_scales.reshape(-1)),
            object_root_position_offsets_w=wp.array(
                root_offsets.reshape(-1, 3),
                dtype=wp.vec3,
                device=device,
            ),
            object_radius=f32((body_scales * radius[None, :]).reshape(-1)),
            voc_object_body_offsets=i32(layout.voc_object_body_offsets),
            voc_object_body_ids=i32(layout.voc_object_body_ids),
            articulation_q_ids=i32(layout.articulation_q_ids),
            articulation_dof_ids=i32(layout.articulation_dof_ids),
            articulation_reference_ids=i32(layout.articulation_reference_ids),
            articulation_kp=f32(layout.articulation_kp),
            articulation_kd=f32(layout.articulation_kd),
            articulation_effort_limit=f32(layout.articulation_effort_limit),
            body_target_pos_w=wp.zeros(scene.world_count * layout.num_bodies, dtype=wp.vec3, device=device),
            body_target_quat_w=wp.zeros(scene.world_count * layout.num_bodies, dtype=wp.quat, device=device),
            voc_target_pos_w=wp.zeros(scene.world_count * layout.num_objects, dtype=wp.vec3, device=device),
            voc_target_quat_w=wp.zeros(scene.world_count * layout.num_objects, dtype=wp.quat, device=device),
            articulation_target_pos=wp.zeros(
                scene.world_count * layout.num_articulations,
                dtype=wp.float32,
                device=device,
            ),
        )

    def gather(self, timestep: wp.array) -> None:
        """Gather each world's current object targets into persistent device buffers."""
        layout = self.layout
        wp.launch(
            gather_object_command,
            dim=self.world_count,
            inputs=[
                timestep,
                self.reference.num_frames,
                layout.num_bodies,
                layout.num_objects,
                layout.num_articulations,
                self.reference.num_articulations,
                self.reference.body_pos_w,
                self.reference.body_quat_w,
                self.reference.articulation_joint_pos,
                self.body_object_ids,
                self.voc_reference_body_ids,
                self.articulation_reference_ids,
                self.object_scales,
                self.object_root_position_offsets_w,
            ],
            outputs=[
                self.body_target_pos_w,
                self.body_target_quat_w,
                self.voc_target_pos_w,
                self.voc_target_quat_w,
                self.articulation_target_pos,
            ],
        )
