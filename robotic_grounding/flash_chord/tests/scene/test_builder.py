# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scene assembly (embodiment + objects + replication)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

pytestmark = pytest.mark.gpu

_HOT3D = (
    ASSETS_DIR
    / "human_motion_data"
    / "hot3d"
    / "hot3d_processed"
    / "sequence_id=P0002_59a84a3a_seg025"
    / "robot_name=sharpa_wave"
)


def _scene(world_count, **kwargs):
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.scene.builder import build_scene

    ref = load_mano_sharpa(str(_HOT3D))
    with wp.ScopedDevice("cuda:0"):
        return build_scene(SharpaHands(), ref, world_count=world_count, **kwargs)


@pytest.mark.sequence_data
def test_scene_assembles_and_replicates():
    s1 = _scene(1)
    assert len(s1.objects) == 2  # hot3d multi-object
    assert s1.robot_reference.num_frames == 100
    assert s1.robot_reference.fps == 30.0
    assert s1.robot_reference.num_joint_q == s1.layout.num_joint_q
    assert s1.robot_reference.num_joint_dof == s1.layout.num_joint_dof
    assert s1.robot_reference.hand("left").finger_joint_pos.shape == (100, 22)
    # one world = robot (sharpa: 56 dof, 52 bodies) + 2 free objects (6 dof, 1 body each)
    assert s1.model.joint_dof_count == s1.layout.num_joint_dof + 6 * len(s1.objects)
    assert s1.model.contacts().force is not None
    np.testing.assert_array_equal(s1.object_scales, 1.0)
    np.testing.assert_array_equal(s1.object_root_position_offsets_w, 0.0)

    base_bodies = s1.model.body_count
    base_dof = s1.model.joint_dof_count
    s2 = _scene(2)
    assert s2.model.body_count == 2 * base_bodies  # replication scales bodies + dofs
    assert s2.model.joint_dof_count == 2 * base_dof


@pytest.mark.sequence_data
def test_scene_applies_contact_friction_and_free_object_damping():
    import newton

    scene = _scene(1, contact_friction=0.5, object_free_joint_damping=0.01)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape_flags = scene.model.shape_flags.numpy()
    friction = scene.model.shape_material_mu.numpy()
    active = (shape_flags & collide) != 0
    np.testing.assert_allclose(friction[active], 0.5)

    solver = newton.solvers.SolverMuJoCo(
        scene.model,
        use_mujoco_contacts=True,
        njmax=64,
        nconmax=32,
    )
    damping = solver.mjw_model.dof_damping.numpy()[0]
    for binding in scene.objects:
        np.testing.assert_allclose(damping[list(binding.root.free_dof_ids)], 0.01)


@pytest.mark.sequence_data
def test_object_scale_randomization_scales_geometry_and_inertia_but_not_mass():
    scene = _scene(
        3,
        object_scale_min=0.8,
        object_scale_max=1.2,
        object_scale_seed=7,
    )
    np.testing.assert_array_equal(scene.object_scales[0], 1.0)
    assert np.all((scene.object_scales[1:] >= 0.8) & (scene.object_scales[1:] <= 1.2))
    assert np.any(scene.object_root_position_offsets_w[1:] != 0.0)

    shape_scale = scene.model.shape_scale.numpy()
    body_mass = scene.model.body_mass.numpy()
    body_inertia = scene.model.body_inertia.numpy()
    shapes_per_world = scene.collision_layout.shape_count
    bodies_per_world = scene.model.body_count // scene.world_count
    for object_id, binding in enumerate(scene.objects):
        scale = scene.object_scales[1, object_id]
        for shape in binding.shapes.ids():
            np.testing.assert_allclose(
                shape_scale[shapes_per_world + shape],
                scale * shape_scale[shape],
            )
        for body in binding.bodies:
            np.testing.assert_allclose(
                body_mass[bodies_per_world + body.body_id],
                body_mass[body.body_id],
            )
            np.testing.assert_allclose(
                body_inertia[bodies_per_world + body.body_id],
                scale**2 * body_inertia[body.body_id],
                rtol=2.0e-6,
                atol=1.0e-10,
            )


def test_ground_free_scene_requires_explicit_support_before_building():
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy

    with pytest.raises(ValueError, match="ground-free scenes require an explicit support"):
        build_scene(None, None, collision=CollisionPolicy(ground=False))


@pytest.mark.sequence_data
def test_scene_rejects_semantic_frames_outside_the_replicated_world_layout():
    from flash_chord.embodiments.base import BodyFrame

    scene = _scene(1)
    left = scene.layout.hand("left")
    invalid_body_id = scene.model.body_count
    invalid_left = replace(
        left,
        palm_frame=BodyFrame(left.palm_frame.name, invalid_body_id),
    )
    semantic_frames = tuple(
        invalid_left.palm_frame if frame.name == left.palm_frame.name else frame
        for frame in scene.layout.semantic_frames
    )
    invalid_layout = replace(
        scene.layout,
        hands=tuple(invalid_left if hand.side == "left" else hand for hand in scene.layout.hands),
        semantic_frames=semantic_frames,
    )

    with pytest.raises(ValueError, match="semantic frame body IDs must be below"):
        replace(scene, layout=invalid_layout)


@pytest.mark.sequence_data
def test_scene_rejects_ambiguous_or_undersized_replicated_joint_topology():
    scene = _scene(2)

    def model_with(**counts):
        return SimpleNamespace(
            body_count=scene.model.body_count,
            shape_count=counts.get("shape_count", scene.model.shape_count),
            joint_coord_count=counts.get("joint_coord_count", scene.model.joint_coord_count),
            joint_dof_count=counts.get("joint_dof_count", scene.model.joint_dof_count),
        )

    with pytest.raises(ValueError, match="scene model has .* shapes; expected"):
        replace(scene, model=model_with(shape_count=scene.model.shape_count - 1))
    with pytest.raises(ValueError, match="joint coordinate count .* not divisible"):
        replace(scene, model=model_with(joint_coord_count=scene.model.joint_coord_count - 1))
    with pytest.raises(ValueError, match="joint DOF count .* not divisible"):
        replace(scene, model=model_with(joint_dof_count=scene.model.joint_dof_count - 1))
    with pytest.raises(ValueError, match="robot layout requires"):
        replace(scene, model=model_with(joint_coord_count=2 * (scene.layout.num_joint_q - 1)))
    with pytest.raises(ValueError, match="robot layout requires"):
        replace(scene, model=model_with(joint_dof_count=2 * (scene.layout.num_joint_dof - 1)))


@pytest.mark.sequence_data
def test_scene_rejects_object_bindings_outside_model_or_collision_ownership():
    from flash_chord.scene.collision import ShapeSpan

    scene = _scene(1)
    first = scene.objects[0]
    q_per_world = scene.model.joint_coord_count
    invalid_root = replace(first.root, free_q_ids=tuple(range(q_per_world, q_per_world + 7)))
    with pytest.raises(ValueError, match="invalid or overlapping q IDs"):
        replace(scene, objects=[replace(first, root=invalid_root), *scene.objects[1:]])

    assert len(first.shapes) > 1
    incomplete_shapes = ShapeSpan(first.shapes.start + 1, first.shapes.stop)
    with pytest.raises(ValueError, match="cover the scene object shape span exactly"):
        replace(scene, objects=[replace(first, shapes=incomplete_shapes), *scene.objects[1:]])
