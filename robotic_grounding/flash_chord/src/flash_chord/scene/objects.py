# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and import explicit reference-object asset bindings into Newton."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import newton
import numpy as np

from flash_chord.data.reference import ObjectAssetSpec, ObjectJointDriveSpec, Reference
from flash_chord.scene.collision import ShapeSpan
from flash_chord.utils.quat import wxyz_to_xyzw


@dataclass(frozen=True)
class ObjectBodyBinding:
    """One reference object body mapped to one imported Newton body."""

    body_id: int
    reference_body_id: int

    def __post_init__(self) -> None:
        if self.body_id < 0 or self.reference_body_id < 0:
            raise ValueError("object body binding IDs must be nonnegative")


@dataclass(frozen=True)
class ObjectRootBinding:
    """The body and exact coordinates/DOFs owned by one object's free root."""

    body_id: int
    reference_body_id: int
    free_q_ids: tuple[int, ...]
    free_dof_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_q_ids", tuple(self.free_q_ids))
        object.__setattr__(self, "free_dof_ids", tuple(self.free_dof_ids))
        if self.body_id < 0 or self.reference_body_id < 0:
            raise ValueError("object root binding IDs must be nonnegative")
        if (
            len(self.free_q_ids) != 7
            or self.free_q_ids != tuple(range(self.free_q_ids[0], self.free_q_ids[0] + 7))
            or self.free_q_ids[0] < 0
        ):
            raise ValueError("object root must bind exactly seven contiguous nonnegative free-joint q IDs")
        if (
            len(self.free_dof_ids) != 6
            or self.free_dof_ids != tuple(range(self.free_dof_ids[0], self.free_dof_ids[0] + 6))
            or self.free_dof_ids[0] < 0
        ):
            raise ValueError("object root must bind exactly six contiguous nonnegative free-joint DOF IDs")


@dataclass(frozen=True)
class ObjectJointBinding:
    """One scalar reference articulation and its implicit target-drive parameters."""

    q_id: int
    dof_id: int
    reference_id: int
    drive: ObjectJointDriveSpec

    def __post_init__(self) -> None:
        if self.q_id < 0 or self.dof_id < 0 or self.reference_id < 0:
            raise ValueError("object articulation binding IDs must be nonnegative")


@dataclass(frozen=True)
class ObjectBinding:
    """Resolved simulation IDs for one independently imported object component."""

    name: str
    bodies: tuple[ObjectBodyBinding, ...]
    root: ObjectRootBinding
    articulations: tuple[ObjectJointBinding, ...]
    shapes: ShapeSpan

    def __post_init__(self) -> None:
        object.__setattr__(self, "bodies", tuple(self.bodies))
        object.__setattr__(self, "articulations", tuple(self.articulations))
        if not self.name or not self.bodies:
            raise ValueError("object bindings require a nonempty name and at least one mapped body")
        body_pairs = tuple((binding.body_id, binding.reference_body_id) for binding in self.bodies)
        if len({body_id for body_id, _ in body_pairs}) != len(body_pairs):
            raise ValueError(f"object binding {self.name!r} maps a simulation body more than once")
        if len({reference_id for _, reference_id in body_pairs}) != len(body_pairs):
            raise ValueError(f"object binding {self.name!r} maps a reference body more than once")
        if (self.root.body_id, self.root.reference_body_id) not in body_pairs:
            raise ValueError(f"object binding {self.name!r} root must match one of its body mappings")
        articulation_q_ids = tuple(binding.q_id for binding in self.articulations)
        articulation_dof_ids = tuple(binding.dof_id for binding in self.articulations)
        articulation_reference_ids = tuple(binding.reference_id for binding in self.articulations)
        if (
            len(set(articulation_q_ids)) != len(articulation_q_ids)
            or len(set(articulation_dof_ids)) != len(articulation_dof_ids)
            or len(set(articulation_reference_ids)) != len(articulation_reference_ids)
        ):
            raise ValueError(f"object binding {self.name!r} must map articulations one-to-one")
        if set(self.root.free_q_ids).intersection(articulation_q_ids):
            raise ValueError(f"object binding {self.name!r} reuses a free-root q ID for an articulation")
        if not len(self.shapes):
            raise ValueError(f"object binding {self.name!r} must own at least one shape")


def _object_builder(destination: newton.ModelBuilder) -> newton.ModelBuilder:
    """Create an isolated object builder with the destination's import defaults."""
    builder = newton.ModelBuilder(up_axis=destination.up_axis, gravity=destination.gravity)
    builder.default_shape_cfg = copy.copy(destination.default_shape_cfg)
    builder.default_joint_cfg = copy.copy(destination.default_joint_cfg)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    return builder


def _append_object(
    destination: newton.ModelBuilder,
    source: newton.ModelBuilder,
    binding: ObjectBinding,
) -> ObjectBinding:
    """Append one isolated component and translate all local binding IDs."""
    body_offset = destination.body_count
    q_offset = destination.joint_coord_count
    dof_offset = destination.joint_dof_count
    shape_offset = destination.shape_count
    destination.add_builder(source)
    if (
        destination.body_count != body_offset + source.body_count
        or destination.joint_coord_count != q_offset + source.joint_coord_count
        or destination.joint_dof_count != dof_offset + source.joint_dof_count
        or destination.shape_count != shape_offset + source.shape_count
    ):
        raise RuntimeError(f"appending object {binding.name!r} did not preserve its component topology")
    return ObjectBinding(
        name=binding.name,
        bodies=tuple(ObjectBodyBinding(body_offset + body.body_id, body.reference_body_id) for body in binding.bodies),
        root=ObjectRootBinding(
            body_id=body_offset + binding.root.body_id,
            reference_body_id=binding.root.reference_body_id,
            free_q_ids=tuple(q_offset + q_id for q_id in binding.root.free_q_ids),
            free_dof_ids=tuple(dof_offset + dof_id for dof_id in binding.root.free_dof_ids),
        ),
        articulations=tuple(
            ObjectJointBinding(
                q_id=q_offset + articulation.q_id,
                dof_id=dof_offset + articulation.dof_id,
                reference_id=articulation.reference_id,
                drive=articulation.drive,
            )
            for articulation in binding.articulations
        ),
        shapes=ShapeSpan(shape_offset + binding.shapes.start, shape_offset + binding.shapes.stop),
    )


def _decompose_object_collision(builder: newton.ModelBuilder, imported_shapes: ShapeSpan, **coacd_params) -> int:
    """Replace colliding object meshes in ``imported_shapes`` with convex collision hulls."""
    import dataclasses

    from flash_chord.assets.mesh import convex_decompose

    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    mesh_type = int(newton.GeoType.MESH)
    mat_fields = ("ke", "kd", "kf", "ka", "mu", "restitution")
    originals = [
        shape
        for shape in imported_shapes.ids()
        if int(builder.shape_type[shape]) == mesh_type and (int(builder.shape_flags[shape]) & collide)
    ]
    new_hulls: list[int] = []
    for shape in originals:
        source = builder.shape_source[shape]
        vertices = np.asarray(source.vertices)
        faces = np.asarray(source.indices).reshape(-1, 3)
        hulls = convex_decompose(vertices, faces, **coacd_params)
        body = builder.shape_body[shape]
        transform = builder.shape_transform[shape]
        scale = builder.shape_scale[shape]
        material = {
            field: float(getattr(builder, f"shape_material_{field}")[shape])
            for field in mat_fields
            if hasattr(builder, f"shape_material_{field}")
        }
        config = dataclasses.replace(
            builder.default_shape_cfg,
            is_visible=False,
            has_shape_collision=True,
            density=0.0,
            **material,
        )
        for hull in hulls:
            mesh = newton.Mesh(
                hull.vertices.astype(np.float32),
                hull.faces.reshape(-1).astype(np.int32),
                compute_inertia=False,
            )
            new_hulls.append(builder.add_shape_mesh(body=body, xform=transform, mesh=mesh, scale=scale, cfg=config))
        builder.shape_flags[shape] &= ~collide

    existing = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
    for index, shape_a in enumerate(new_hulls):
        for shape_b in new_hulls[index + 1 :]:
            pair = (min(shape_a, shape_b), max(shape_a, shape_b))
            if pair not in existing:
                builder.add_shape_collision_filter_pair(*pair)
                existing.add(pair)
    return len(new_hulls)


def _terminal_ids(labels, requested: tuple[str, ...], *, collection: str) -> dict[str, int]:
    """Resolve exact terminal names inside one isolated imported component."""
    matches: dict[str, list[int]] = {name: [] for name in requested}
    for item_id, label in enumerate(labels):
        terminal = str(label).rsplit("/", 1)[-1]
        if terminal in matches:
            matches[terminal].append(item_id)
    missing = tuple(name for name, ids in matches.items() if not ids)
    ambiguous = {name: tuple(ids) for name, ids in matches.items() if len(ids) > 1}
    if missing or ambiguous:
        raise ValueError(f"unable to resolve object {collection}: missing={missing}, ambiguous={ambiguous}")
    return {name: ids[0] for name, ids in matches.items()}


def _coordinate_width(starts, total: int, item_id: int) -> int:
    """Return one joint's coordinate width even when fixed joints share starts."""
    start = int(starts[item_id])
    stop = min((int(value) for value in starts if int(value) > start), default=total)
    return stop - start


def _resolve_body_bindings(
    builder: newton.ModelBuilder,
    asset: ObjectAssetSpec,
    reference_body_ids: dict[str, int],
) -> tuple[ObjectBodyBinding, ...]:
    named = tuple(body.simulation_name for body in asset.bodies if body.simulation_name is not None)
    named_ids = _terminal_ids(builder.body_label, named, collection="bodies") if named else {}
    if any(body.simulation_name is None for body in asset.bodies) and builder.body_count != 1:
        raise ValueError(
            f"unnamed object body binding in {asset.name!r} requires exactly one imported body, "
            f"got {builder.body_count}"
        )
    bindings = tuple(
        ObjectBodyBinding(
            body_id=0 if body.simulation_name is None else named_ids[body.simulation_name],
            reference_body_id=reference_body_ids[body.reference_name],
        )
        for body in asset.bodies
    )
    mapped_body_ids = {binding.body_id for binding in bindings}
    imported_body_ids = set(range(builder.body_count))
    if mapped_body_ids != imported_body_ids:
        raise ValueError(
            f"object asset {asset.name!r} body declarations must cover every imported body exactly once; "
            f"mapped={tuple(sorted(mapped_body_ids))}, imported={tuple(sorted(imported_body_ids))}"
        )
    return bindings


def _resolve_root(
    builder: newton.ModelBuilder,
    asset: ObjectAssetSpec,
    bodies: tuple[ObjectBodyBinding, ...],
    reference_body_ids: dict[str, int],
) -> ObjectRootBinding:
    root_reference_id = reference_body_ids[asset.root_reference_name]
    root_body_id = next(body.body_id for body in bodies if body.reference_body_id == root_reference_id)
    free_joints = tuple(
        joint_id for joint_id, joint_type in enumerate(builder.joint_type) if joint_type == newton.JointType.FREE
    )
    root_joints = tuple(joint_id for joint_id in free_joints if builder.joint_child[joint_id] == root_body_id)
    if len(free_joints) != 1 or len(root_joints) != 1 or builder.joint_parent[root_joints[0]] != -1:
        raise ValueError(
            f"object asset {asset.name!r} must import exactly one world-parented free joint on its root body; "
            f"free_joints={free_joints}, root_body={root_body_id}, root_joints={root_joints}"
        )
    joint_id = root_joints[0]
    if (
        _coordinate_width(builder.joint_q_start, builder.joint_coord_count, joint_id) != 7
        or _coordinate_width(builder.joint_qd_start, builder.joint_dof_count, joint_id) != 6
    ):
        raise ValueError(f"object asset {asset.name!r} free root does not expose 7 q coordinates and 6 DOFs")
    q_start = int(builder.joint_q_start[joint_id])
    dof_start = int(builder.joint_qd_start[joint_id])
    return ObjectRootBinding(
        root_body_id,
        root_reference_id,
        tuple(range(q_start, q_start + 7)),
        tuple(range(dof_start, dof_start + 6)),
    )


def _resolve_articulations(
    builder: newton.ModelBuilder,
    asset: ObjectAssetSpec,
    bodies: tuple[ObjectBodyBinding, ...],
) -> tuple[ObjectJointBinding, ...]:
    names = tuple(articulation.simulation_joint_name for articulation in asset.articulations)
    joint_ids = _terminal_ids(builder.joint_label, names, collection="joints") if names else {}
    declared = set(joint_ids.values())
    imported_movable = {
        joint_id
        for joint_id, joint_type in enumerate(builder.joint_type)
        if joint_type not in (newton.JointType.FIXED, newton.JointType.FREE)
    }
    if declared != imported_movable:
        undeclared = tuple(sorted(imported_movable - declared))
        invalid = tuple(sorted(declared - imported_movable))
        raise ValueError(
            f"object asset {asset.name!r} articulation declaration does not match imported movable joints: "
            f"undeclared={undeclared}, non_movable={invalid}"
        )

    bindings = []
    mapped_body_ids = {body.body_id for body in bodies}
    for articulation in asset.articulations:
        joint_id = joint_ids[articulation.simulation_joint_name]
        joint_type = builder.joint_type[joint_id]
        if joint_type not in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
            raise ValueError(
                f"object articulation {articulation.simulation_joint_name!r} must be revolute or prismatic, "
                f"got {joint_type}"
            )
        if (
            _coordinate_width(builder.joint_q_start, builder.joint_coord_count, joint_id) != 1
            or _coordinate_width(builder.joint_qd_start, builder.joint_dof_count, joint_id) != 1
        ):
            raise ValueError(f"object articulation {articulation.simulation_joint_name!r} must be scalar")
        parent = int(builder.joint_parent[joint_id])
        child = int(builder.joint_child[joint_id])
        if parent not in mapped_body_ids or child not in mapped_body_ids:
            raise ValueError(
                f"object articulation {articulation.simulation_joint_name!r} connects undeclared bodies "
                f"({parent}, {child})"
            )
        q_id = int(builder.joint_q_start[joint_id])
        dof_id = int(builder.joint_qd_start[joint_id])
        builder.joint_armature[dof_id] = articulation.physics.armature
        builder.joint_friction[dof_id] = articulation.physics.friction
        builder.joint_target_ke[dof_id] = articulation.drive.kp
        builder.joint_target_kd[dof_id] = articulation.drive.kd
        builder.joint_target_mode[dof_id] = int(newton.JointTargetMode.POSITION_VELOCITY)
        builder.joint_effort_limit[dof_id] = articulation.drive.effort_limit
        bindings.append(ObjectJointBinding(q_id, dof_id, articulation.reference_index, articulation.drive))
    return tuple(bindings)


def _validate_connected_component(
    builder: newton.ModelBuilder,
    asset: ObjectAssetSpec,
    root: ObjectRootBinding,
) -> None:
    """Require every imported body to belong to the mechanism rooted at the declared free body."""
    reachable = {root.body_id}
    changed = True
    while changed:
        changed = False
        for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True):
            parent = int(parent)
            child = int(child)
            if parent in reachable and child >= 0 and child not in reachable:
                reachable.add(child)
                changed = True
    imported = set(range(builder.body_count))
    if reachable != imported:
        raise ValueError(
            f"object asset {asset.name!r} must be one mechanism rooted at body {root.body_id}; "
            f"unreachable={tuple(sorted(imported - reachable))}"
        )


def _spawn_component(
    destination: newton.ModelBuilder,
    asset: ObjectAssetSpec,
    reference_body_ids: dict[str, int],
    body_pos0: np.ndarray,
    body_quat0: np.ndarray,
    articulation0: np.ndarray,
    *,
    decompose_collision: bool,
    decompose_params: dict,
) -> tuple[ObjectBinding, newton.ModelBuilder]:
    component = _object_builder(destination)
    component.add_urdf(
        source=asset.urdf_path,
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    imported_shapes = ShapeSpan(0, component.shape_count)
    bodies = _resolve_body_bindings(component, asset, reference_body_ids)
    root = _resolve_root(component, asset, bodies, reference_body_ids)
    articulations = _resolve_articulations(component, asset, bodies)
    _validate_connected_component(component, asset, root)

    root_position = body_pos0[root.reference_body_id]
    root_quaternion = wxyz_to_xyzw(body_quat0[root.reference_body_id])
    q_start = root.free_q_ids[0]
    component.joint_q[q_start : q_start + 3] = [float(value) for value in root_position]
    component.joint_q[q_start + 3 : q_start + 7] = [float(value) for value in root_quaternion]
    for articulation in articulations:
        target = float(articulation0[articulation.reference_id])
        component.joint_q[articulation.q_id] = target

    if decompose_collision:
        _decompose_object_collision(component, imported_shapes, **decompose_params)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    if not any(int(component.shape_flags[shape]) & collide for shape in range(component.shape_count)):
        raise ValueError(f"object asset {asset.name!r} does not provide any active collision shapes")
    return (
        ObjectBinding(
            name=asset.name,
            bodies=bodies,
            root=root,
            articulations=articulations,
            shapes=ShapeSpan(0, component.shape_count),
        ),
        component,
    )


def _validate_reference_contract(reference: Reference) -> tuple[
    tuple[ObjectAssetSpec, ...],
    dict[str, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    names = tuple(reference.object_body_names())
    body_pos = np.asarray(reference.object_body_pos_w())
    body_quat = np.asarray(reference.object_body_quat_w())
    articulation = np.asarray(reference.object_articulation())
    expected_pos_shape = (reference.num_frames, len(names), 3)
    expected_quat_shape = (reference.num_frames, len(names), 4)
    if not names or len(set(names)) != len(names):
        raise ValueError("reference object body names must be nonempty and unique")
    if body_pos.shape != expected_pos_shape or body_quat.shape != expected_quat_shape:
        raise ValueError(
            f"object pose arrays must have shapes {expected_pos_shape} and {expected_quat_shape}; "
            f"got {body_pos.shape} and {body_quat.shape}"
        )
    if articulation.ndim != 2 or articulation.shape[0] != reference.num_frames:
        raise ValueError(f"object articulation must have shape ({reference.num_frames}, A), got {articulation.shape}")
    if not np.all(np.isfinite(body_pos)) or not np.all(np.isfinite(body_quat)) or not np.all(np.isfinite(articulation)):
        raise ValueError("object reference trajectories must contain only finite values")
    quaternion_norm = np.linalg.norm(body_quat, axis=-1)
    if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("object reference quaternions must be unit length")

    assets = tuple(reference.object_assets())
    if not assets or len({asset.name for asset in assets}) != len(assets):
        raise ValueError("object asset names must be nonempty and unique")
    declared_bodies = tuple(body.reference_name for asset in assets for body in asset.bodies)
    declared_articulations = tuple(joint.reference_index for asset in assets for joint in asset.articulations)
    if len(declared_bodies) != len(set(declared_bodies)) or set(declared_bodies) != set(names):
        raise ValueError(
            "object asset body declarations must cover every reference body exactly once; "
            f"reference={names}, declared={declared_bodies}"
        )
    expected_articulations = set(range(articulation.shape[1]))
    if (
        len(declared_articulations) != len(set(declared_articulations))
        or set(declared_articulations) != expected_articulations
    ):
        raise ValueError(
            "object asset articulation declarations must cover every reference column exactly once; "
            f"reference={tuple(sorted(expected_articulations))}, declared={declared_articulations}"
        )
    return assets, {name: index for index, name in enumerate(names)}, body_pos, body_quat, articulation


def spawn_objects(
    builder: newton.ModelBuilder,
    reference: Reference,
    *,
    decompose_collision: bool = True,
    decompose_params: dict | None = None,
) -> list[ObjectBinding]:
    """Import reference-declared object assets and return exact resolved simulation bindings."""
    assets, reference_body_ids, body_pos, body_quat, articulation = _validate_reference_contract(reference)
    bindings = []
    for asset in assets:
        local_binding, component = _spawn_component(
            builder,
            asset,
            reference_body_ids,
            body_pos[0],
            body_quat[0],
            articulation[0],
            decompose_collision=decompose_collision,
            decompose_params={} if decompose_params is None else decompose_params,
        )
        bindings.append(_append_object(builder, component, local_binding))
    return bindings
