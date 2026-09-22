# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a scene: embodiment + reference objects, replicated to N worlds."""

from __future__ import annotations

from dataclasses import dataclass, replace

import newton
import numpy as np

from flash_chord.data.reference import Reference
from flash_chord.embodiments.base import Embodiment, EmbodimentLayout
from flash_chord.embodiments.binding import RobotReferenceBinding
from flash_chord.scene.collision import (
    CollisionPolicy,
    SceneCollisionLayout,
    ShapeSpan,
    apply_collision_policy,
    capture_collision_layout,
)
from flash_chord.scene.objects import ObjectBinding, spawn_objects
from flash_chord.scene.support import SupportBinding, add_support_surfaces
from flash_chord.utils.quat import quat_rotate_xyzw, wxyz_to_xyzw


@dataclass
class Scene:
    model: newton.Model
    layout: EmbodimentLayout  # per-world embodiment DOF/body layout
    robot_reference: RobotReferenceBinding  # immutable simulation-ordered robot trajectories
    collision_layout: SceneCollisionLayout  # per-copy shape ownership before replication
    collision_policy: CollisionPolicy
    objects: list[ObjectBinding]  # resolved per-world object bindings
    support: SupportBinding | None  # resolved per-world support component, when configured
    object_scales: np.ndarray  # fixed isotropic scale per world and logical object [W, O]
    object_root_position_offsets_w: np.ndarray  # bottom-height-preserving root offsets [W, O, 3]
    world_count: int

    def __post_init__(self) -> None:
        if self.world_count <= 0:
            raise ValueError(f"scene world count must be positive, got {self.world_count}")
        if self.object_scales.shape != (self.world_count, len(self.objects)):
            raise ValueError(
                f"object scales have shape {self.object_scales.shape}; "
                f"expected ({self.world_count}, {len(self.objects)})"
            )
        if self.object_root_position_offsets_w.shape != (
            self.world_count,
            len(self.objects),
            3,
        ):
            raise ValueError(
                f"object root offsets have shape {self.object_root_position_offsets_w.shape}; "
                f"expected ({self.world_count}, {len(self.objects)}, 3)"
            )
        if not np.all(np.isfinite(self.object_scales)) or np.any(self.object_scales <= 0.0):
            raise ValueError("object scales must be finite and positive")
        if not np.all(np.isfinite(self.object_root_position_offsets_w)):
            raise ValueError("object root offsets must be finite")
        self.robot_reference.validate(self.layout)
        if self.model.body_count % self.world_count:
            raise ValueError(
                f"scene body count {self.model.body_count} is not divisible by world count {self.world_count}"
            )
        expected_shape_count = self.world_count * self.collision_layout.shape_count + int(self.collision_policy.ground)
        if self.model.shape_count != expected_shape_count:
            raise ValueError(
                f"scene model has {self.model.shape_count} shapes; expected {expected_shape_count} from "
                "replicated collision ownership and ground policy"
            )
        topology = (
            ("joint coordinate", self.model.joint_coord_count, self.layout.num_joint_q),
            ("joint DOF", self.model.joint_dof_count, self.layout.num_joint_dof),
        )
        for label, total, robot_count in topology:
            if total % self.world_count:
                raise ValueError(f"scene {label} count {total} is not divisible by world count {self.world_count}")
            per_world = total // self.world_count
            if per_world < robot_count:
                raise ValueError(f"scene has {per_world} {label}s per world; robot layout requires {robot_count}")
        bodies_per_world = self.model.body_count // self.world_count
        q_per_world = self.model.joint_coord_count // self.world_count
        dof_per_world = self.model.joint_dof_count // self.world_count
        _validate_component_bindings(
            self.objects,
            self.support,
            self.collision_layout.objects,
            self.collision_layout.support,
            bodies_per_world=bodies_per_world,
            q_per_world=q_per_world,
            dof_per_world=dof_per_world,
            robot_q_count=self.layout.num_joint_q,
            robot_dof_count=self.layout.num_joint_dof,
        )
        invalid_frames = tuple(
            (frame.name, frame.body_id) for frame in self.layout.semantic_frames if frame.body_id >= bodies_per_world
        )
        if invalid_frames:
            raise ValueError(f"semantic frame body IDs must be below {bodies_per_world}, got {invalid_frames}")

    @property
    def support_shape_count(self) -> int:
        """Number of support shapes in one unreplicated scene copy."""
        return 0 if self.support is None else len(self.support.shapes)


def _validate_component_bindings(
    objects: list[ObjectBinding],
    support: SupportBinding | None,
    object_shapes: ShapeSpan,
    support_shapes: ShapeSpan,
    *,
    bodies_per_world: int,
    q_per_world: int,
    dof_per_world: int,
    robot_q_count: int,
    robot_dof_count: int,
) -> None:
    """Validate resolved object IDs against one unreplicated scene copy."""
    if len({binding.name for binding in objects}) != len(objects):
        raise ValueError("scene object binding names must be unique")
    simulation_body_ids: set[int] = set()
    reference_body_ids: set[int] = set()
    object_q_ids: set[int] = set()
    object_dof_ids: set[int] = set()
    reference_articulation_ids: set[int] = set()
    assigned_shapes: set[int] = set()
    for binding in objects:
        for body in binding.bodies:
            if body.body_id >= bodies_per_world:
                raise ValueError(f"object {binding.name!r} body ID {body.body_id} must be below {bodies_per_world}")
            if body.body_id in simulation_body_ids or body.reference_body_id in reference_body_ids:
                raise ValueError(f"scene object {binding.name!r} duplicates a body mapping")
            simulation_body_ids.add(body.body_id)
            reference_body_ids.add(body.reference_body_id)
        q_ids = binding.root.free_q_ids + tuple(joint.q_id for joint in binding.articulations)
        invalid_q = tuple(q_id for q_id in q_ids if not robot_q_count <= q_id < q_per_world)
        overlap_q = object_q_ids.intersection(q_ids)
        if invalid_q or overlap_q:
            raise ValueError(
                f"object {binding.name!r} has invalid or overlapping q IDs: "
                f"invalid={invalid_q}, overlap={tuple(sorted(overlap_q))}"
            )
        object_q_ids.update(q_ids)
        dof_ids = binding.root.free_dof_ids + tuple(joint.dof_id for joint in binding.articulations)
        articulation_reference_ids = tuple(joint.reference_id for joint in binding.articulations)
        overlap_reference_articulations = reference_articulation_ids.intersection(articulation_reference_ids)
        invalid_dof = tuple(dof_id for dof_id in dof_ids if not robot_dof_count <= dof_id < dof_per_world)
        overlap_dof = object_dof_ids.intersection(dof_ids)
        if invalid_dof or overlap_dof or overlap_reference_articulations:
            raise ValueError(
                f"object {binding.name!r} has invalid or overlapping DOF IDs: "
                f"invalid={invalid_dof}, overlap={tuple(sorted(overlap_dof))}, "
                f"reference_overlap={tuple(sorted(overlap_reference_articulations))}"
            )
        object_dof_ids.update(dof_ids)
        reference_articulation_ids.update(articulation_reference_ids)
        shape_ids = set(binding.shapes.ids())
        invalid_shapes = shape_ids - set(object_shapes.ids())
        overlap_shapes = assigned_shapes.intersection(shape_ids)
        if invalid_shapes or overlap_shapes:
            raise ValueError(
                f"object {binding.name!r} has invalid or overlapping shape ownership: "
                f"invalid={tuple(sorted(invalid_shapes))}, overlap={tuple(sorted(overlap_shapes))}"
            )
        assigned_shapes.update(shape_ids)
    if reference_body_ids != set(range(len(reference_body_ids))):
        raise ValueError(f"scene reference object body IDs must be contiguous from zero, got {reference_body_ids}")
    if reference_articulation_ids != set(range(len(reference_articulation_ids))):
        raise ValueError(
            f"scene reference object articulation IDs must be contiguous from zero, got {reference_articulation_ids}"
        )
    if support is None:
        if len(support_shapes):
            raise ValueError("scene collision layout owns support shapes without a support binding")
    else:
        if support.body_id >= bodies_per_world or support.body_id in simulation_body_ids:
            raise ValueError(f"support body ID {support.body_id} is invalid or overlaps an object body")
        support_q_ids = set(support.free_q_ids)
        support_dof_ids = set(support.free_dof_ids)
        invalid_support_q = tuple(q_id for q_id in support.free_q_ids if not robot_q_count <= q_id < q_per_world)
        invalid_support_dof = tuple(
            dof_id for dof_id in support.free_dof_ids if not robot_dof_count <= dof_id < dof_per_world
        )
        if invalid_support_q or object_q_ids.intersection(support_q_ids):
            raise ValueError(
                f"support has invalid or overlapping q IDs: invalid={invalid_support_q}, "
                f"overlap={tuple(sorted(object_q_ids.intersection(support_q_ids)))}"
            )
        if invalid_support_dof or object_dof_ids.intersection(support_dof_ids):
            raise ValueError(
                f"support has invalid or overlapping DOF IDs: invalid={invalid_support_dof}, "
                f"overlap={tuple(sorted(object_dof_ids.intersection(support_dof_ids)))}"
            )
        if support.shapes != support_shapes:
            raise ValueError(
                f"support binding shapes {support.shapes} do not match collision ownership {support_shapes}"
            )
        object_q_ids.update(support_q_ids)
        object_dof_ids.update(support_dof_ids)
    expected_q_ids = set(range(robot_q_count, q_per_world))
    if object_q_ids != expected_q_ids:
        raise ValueError(
            "scene object/support bindings must cover every post-robot q coordinate exactly; "
            f"missing={tuple(sorted(expected_q_ids - object_q_ids))}, "
            f"extra={tuple(sorted(object_q_ids - expected_q_ids))}"
        )
    expected_dof_ids = set(range(robot_dof_count, dof_per_world))
    if object_dof_ids != expected_dof_ids:
        raise ValueError(
            "scene object/support bindings must cover every post-robot DOF exactly; "
            f"missing={tuple(sorted(expected_dof_ids - object_dof_ids))}, "
            f"extra={tuple(sorted(object_dof_ids - expected_dof_ids))}"
        )
    expected_shapes = set(object_shapes.ids())
    if assigned_shapes != expected_shapes:
        raise ValueError(
            "object bindings must cover the scene object shape span exactly; "
            f"missing={tuple(sorted(expected_shapes - assigned_shapes))}"
        )


def _set_object_contact_params(
    builder,
    object_shapes: ShapeSpan,
    solimp,
    priority: int,
    ke=None,
) -> int:
    """Stiffen the object shapes' MuJoCo contact: per-shape ``geom_solimp`` (impedance) + ``geom_priority``
    (so the object's params govern every hand↔object contact, regardless of the hand geoms) + optionally the
    object ``ke`` (which sets the contact ``solref`` stiffness). ``solimp`` is a 5-tuple
    ``(dmin, dmax, width, midpoint, power)``; raising dmin/dmax toward 1 enforces the contact, and ``ke``
    stiffens the underlying spring (object-only, so the global solver stays stable)."""
    solimp_attr = builder.custom_attributes["mujoco:geom_solimp"]
    priority_attr = builder.custom_attributes["mujoco:geom_priority"]
    n = 0
    for shape in object_shapes.ids():
        if solimp is not None:
            solimp_attr.values[shape] = tuple(float(x) for x in solimp)
        if priority:
            priority_attr.values[shape] = int(priority)
        if ke is not None:
            builder.shape_material_ke[shape] = float(ke)
        n += 1
    return n


def _apply_scene_physics_parity(
    builder: newton.ModelBuilder,
    objects: list[ObjectBinding],
    *,
    contact_friction: float,
    object_free_joint_damping: float,
) -> None:
    """Apply backend-representable physical material and rigid-object damping values."""
    for name, value in (
        ("contact_friction", contact_friction),
        ("object_free_joint_damping", object_free_joint_damping),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")

    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for shape in range(builder.shape_count):
        if int(builder.shape_flags[shape]) & collide:
            builder.shape_material_mu[shape] = float(contact_friction)

    damping = builder.custom_attributes.get("mujoco:dof_passive_damping")
    if damping is None:
        raise ValueError("scene physics parity requires the MuJoCo passive-damping attribute")
    for binding in objects:
        for dof_id in binding.root.free_dof_ids:
            damping.values[dof_id] = float(object_free_joint_damping)


def _sample_object_scales(
    world_count: int,
    object_count: int,
    minimum: float,
    maximum: float,
    seed: int,
) -> np.ndarray:
    """Draw one fixed isotropic scale per world/object, keeping world zero nominal."""
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum <= 0.0 or maximum < minimum:
        raise ValueError(f"invalid object scale range [{minimum}, {maximum}]")
    if minimum == maximum:
        return np.full((world_count, object_count), minimum, dtype=np.float32)
    scales = np.random.default_rng(seed).uniform(
        minimum,
        maximum,
        size=(world_count, object_count),
    )
    scales = scales.astype(np.float32)
    if minimum <= 1.0 <= maximum:
        scales[0] = 1.0
    return scales


def _object_lowest_world_z(
    builder: newton.ModelBuilder,
    binding: ObjectBinding,
    body_pos0: np.ndarray,
    body_quat0_wxyz: np.ndarray,
) -> float:
    """Return the lowest active collision vertex at reference frame zero."""
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    mesh_type = int(newton.GeoType.MESH)
    reference_body_by_sim = {body.body_id: body.reference_body_id for body in binding.bodies}
    lowest = np.inf
    for shape in binding.shapes.ids():
        if not (int(builder.shape_flags[shape]) & collide):
            continue
        if int(builder.shape_type[shape]) != mesh_type:
            raise ValueError(
                "object scaling currently requires mesh collision shapes; "
                f"object {binding.name!r} has shape {shape} of type "
                f"{builder.shape_type[shape]}"
            )
        source = builder.shape_source[shape]
        if source is None or not hasattr(source, "vertices"):
            raise ValueError(f"object {binding.name!r} collision shape {shape} has no mesh vertices")
        body_id = int(builder.shape_body[shape])
        reference_body_id = reference_body_by_sim[body_id]
        vertices = np.asarray(source.vertices, dtype=np.float64)
        shape_scale = np.asarray(builder.shape_scale[shape], dtype=np.float64)
        transform = builder.shape_transform[shape]
        shape_position = np.asarray([float(transform[index]) for index in range(3)])
        shape_quaternion = np.asarray([float(transform[index]) for index in range(3, 7)])
        points_body = quat_rotate_xyzw(shape_quaternion, vertices * shape_scale) + shape_position
        body_quaternion = wxyz_to_xyzw(body_quat0_wxyz[reference_body_id])
        points_world = quat_rotate_xyzw(body_quaternion, points_body) + body_pos0[reference_body_id]
        lowest = min(lowest, float(points_world[:, 2].min()))
    if not np.isfinite(lowest):
        raise ValueError(f"object {binding.name!r} has no active mesh collision vertices")
    return lowest


def _scale_transform_translation(transform, scale: float):
    values = [float(transform[index]) for index in range(7)]
    values[:3] = [scale * value for value in values[:3]]
    return type(transform)(*values)


def _apply_object_scales(
    builder: newton.ModelBuilder,
    copy: newton.ModelBuilder,
    objects: list[ObjectBinding],
    reference: Reference,
    scales: np.ndarray,
    *,
    preserve_bottom_height: bool,
) -> np.ndarray:
    """Scale replicated object geometry and inertia while keeping mass fixed."""
    world_count, object_count = scales.shape
    root_offsets = np.zeros((world_count, object_count, 3), dtype=np.float32)
    if np.allclose(scales, 1.0):
        return root_offsets

    body_pos0 = np.asarray(reference.object_body_pos_w()[0], dtype=np.float64)
    body_quat0 = np.asarray(reference.object_body_quat_w()[0], dtype=np.float64)
    if preserve_bottom_height:
        for object_id, binding in enumerate(objects):
            lowest_z = _object_lowest_world_z(
                copy,
                binding,
                body_pos0,
                body_quat0,
            )
            root_z = float(body_pos0[binding.root.reference_body_id, 2])
            root_offsets[:, object_id, 2] = (1.0 - scales[:, object_id]) * (lowest_z - root_z)

    body_count = copy.body_count
    shape_count = copy.shape_count
    joint_count = copy.joint_count
    q_count = copy.joint_coord_count
    for world in range(world_count):
        for object_id, binding in enumerate(objects):
            scale = float(scales[world, object_id])
            offset = root_offsets[world, object_id]
            body_ids = {body.body_id for body in binding.bodies}
            for shape in binding.shapes.ids():
                shape_id = world * shape_count + shape
                builder.shape_scale[shape_id] = tuple(scale * float(value) for value in builder.shape_scale[shape_id])
                builder.shape_transform[shape_id] = _scale_transform_translation(
                    builder.shape_transform[shape_id],
                    scale,
                )
            for body in body_ids:
                body_id = world * body_count + body
                center = builder.body_com[body_id]
                builder.body_com[body_id] = type(center)(*(scale * float(center[index]) for index in range(3)))
                inertia = builder.body_inertia[body_id]
                builder.body_inertia[body_id] = type(inertia)(
                    *(scale * scale * float(inertia[row, column]) for row in range(3) for column in range(3))
                )
            for joint in range(joint_count):
                parent = int(copy.joint_parent[joint])
                child = int(copy.joint_child[joint])
                if parent not in body_ids or child not in body_ids:
                    continue
                if copy.joint_type[joint] == newton.JointType.PRISMATIC:
                    raise ValueError("isotropic object scaling does not support prismatic object joints")
                joint_id = world * joint_count + joint
                builder.joint_X_p[joint_id] = _scale_transform_translation(
                    builder.joint_X_p[joint_id],
                    scale,
                )
                builder.joint_X_c[joint_id] = _scale_transform_translation(
                    builder.joint_X_c[joint_id],
                    scale,
                )
            q_base = world * q_count + binding.root.free_q_ids[0]
            for axis in range(3):
                builder.joint_q[q_base + axis] += float(offset[axis])
    return root_offsets


def build_scene(
    embodiment: Embodiment,
    reference: Reference,
    world_count: int = 1,
    support_usda=None,
    collision: CollisionPolicy | None = None,
    object_contact_solimp=None,
    object_contact_priority: int = 0,
    object_contact_ke=None,
    decompose_objects: bool = True,
    decompose_params: dict | None = None,
    object_scale_min: float = 1.0,
    object_scale_max: float = 1.0,
    object_scale_seed: int = 0,
    contact_friction: float = 1.0,
    object_free_joint_damping: float = 0.0,
) -> Scene:
    """Build one copy (embodiment + objects + support), replicate to ``world_count``, and finalize.

    The robot is at its default pose; objects are at reference frame 0 (the robot is posed
    to the reference at reset). ``layout``/``objects`` ids are per-copy (pre-replication).
    Component shape ownership is captured at append time and one typed collision policy is applied before
    replication. ``decompose_objects`` convex-decomposes each object's collision mesh (CoACD) into multiple
    convex hulls so concave objects contact correctly; ``decompose_params`` overrides CoACD parameters.
    """
    collision = collision or CollisionPolicy()
    if not collision.ground and support_usda is None:
        raise ValueError("ground-free scenes require an explicit support surface")
    copy = newton.ModelBuilder()
    robot_shape_start = len(copy.shape_body)
    layout = embodiment.build(copy)  # registers custom attrs + adds the robot + control config
    robot_shapes = ShapeSpan(robot_shape_start, len(copy.shape_body))
    robot_reference = embodiment.bind_reference(copy, layout, reference)
    _validate_robot_reference(robot_reference, layout, reference)
    object_shape_start = len(copy.shape_body)
    objects = spawn_objects(  # adds objects into the same copy (collision meshes convex-decomposed)
        copy, reference, decompose_collision=decompose_objects, decompose_params=decompose_params
    )
    object_shapes = ShapeSpan(object_shape_start, len(copy.shape_body))
    support_shape_start = len(copy.shape_body)
    support = add_support_surfaces(copy, support_usda) if support_usda is not None else None
    support_shapes = ShapeSpan(support_shape_start, len(copy.shape_body))
    if support is not None and support.shapes != support_shapes:
        raise ValueError(f"support loader bound {support.shapes} but appended scene span {support_shapes}")
    collision_layout = capture_collision_layout(
        copy,
        layout,
        robot_shapes,
        object_shapes,
        support_shapes,
    )
    apply_collision_policy(copy, collision_layout, collision)
    _apply_scene_physics_parity(
        copy,
        objects,
        contact_friction=contact_friction,
        object_free_joint_damping=object_free_joint_damping,
    )
    if object_contact_solimp is not None or object_contact_priority or object_contact_ke is not None:
        _set_object_contact_params(
            copy,
            object_shapes,
            object_contact_solimp,
            object_contact_priority,
            object_contact_ke,
        )
    copy.request_contact_attributes("force")

    builder = newton.ModelBuilder()
    builder.replicate(copy, world_count)
    object_scales = _sample_object_scales(
        world_count,
        len(objects),
        object_scale_min,
        object_scale_max,
        object_scale_seed,
    )
    object_root_offsets = _apply_object_scales(
        builder,
        copy,
        objects,
        reference,
        object_scales,
        preserve_bottom_height=support is not None or collision.ground,
    )
    if collision.ground:
        builder.add_ground_plane(cfg=replace(builder.default_shape_cfg, mu=float(contact_friction)))
    model = builder.finalize()
    return Scene(
        model=model,
        layout=layout,
        robot_reference=robot_reference,
        collision_layout=collision_layout,
        collision_policy=collision,
        objects=objects,
        support=support,
        object_scales=object_scales,
        object_root_position_offsets_w=object_root_offsets,
        world_count=world_count,
    )


def _validate_robot_reference(
    binding: RobotReferenceBinding,
    layout: EmbodimentLayout,
    reference: Reference,
) -> None:
    binding.validate(layout)
    if binding.num_frames != reference.num_frames:
        raise ValueError(f"robot binding has {binding.num_frames} frames but task reference has {reference.num_frames}")
    if binding.fps != reference.fps:
        raise ValueError(f"robot binding has {binding.fps} fps but task reference has {reference.fps} fps")
