# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reference object spawning."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

pytestmark = [pytest.mark.gpu, pytest.mark.sequence_data]

_HOT3D = (
    ASSETS_DIR
    / "human_motion_data"
    / "hot3d"
    / "hot3d_processed"
    / "sequence_id=P0002_59a84a3a_seg025"
    / "robot_name=sharpa_wave"
)
_ARCTIC_BOX = (
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s07_box_grab_01"
    / "robot_name=sharpa_wave"
)


def _with_object_assets(reference, assets):
    """Expose only the object scene-construction contract with replaced asset specs."""
    return SimpleNamespace(
        num_frames=reference.num_frames,
        object_body_names=reference.object_body_names,
        object_body_pos_w=reference.object_body_pos_w,
        object_body_quat_w=reference.object_body_quat_w,
        object_articulation=reference.object_articulation,
        object_assets=lambda: tuple(assets),
    )


def test_hot3d_multi_object_spawn_at_reference_pose():
    import warp as wp

    import newton
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.scene.objects import spawn_objects

    ref = load_mano_sharpa(str(_HOT3D))
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        bindings = spawn_objects(builder, ref)
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()

    assert len(bindings) == 2  # hot3d is two independent rigid components
    pos0 = ref.object_body_pos_w()[0]
    for reference_id, binding in enumerate(bindings):
        assert len(binding.bodies) == 1 and not binding.articulations
        body = binding.bodies[0]
        assert (binding.root.body_id, binding.root.reference_body_id) == (body.body_id, reference_id)
        assert len(binding.root.free_q_ids) == 7 and len(binding.root.free_dof_ids) == 6
        assert body.reference_body_id == reference_id
        assert np.allclose(body_q[body.body_id][:3], pos0[reference_id], atol=1e-3)
    assert bindings[0].shapes.stop <= bindings[1].shapes.start


def test_arctic_articulated_object_spawn_at_reference_pose():
    import warp as wp

    import newton
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.scene.objects import spawn_objects

    ref = load_mano_sharpa(str(_ARCTIC_BOX))
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        bindings = spawn_objects(builder, ref)
        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_contacts=True,
            njmax=64,
            nconmax=32,
        )
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()

    assert len(bindings) == 1
    binding = bindings[0]
    assert tuple((body.body_id, body.reference_body_id) for body in binding.bodies) == ((0, 0), (1, 1))
    assert (binding.root.body_id, binding.root.reference_body_id) == (0, 0)
    assert binding.root.free_q_ids == tuple(range(7))
    assert binding.root.free_dof_ids == tuple(range(6))
    assert model.body_count == 2  # bottom + top
    assert model.joint_dof_count == 7  # 6-DOF free base + 1 revolute articulation
    assert len(binding.articulations) == 1
    articulation = binding.articulations[0]
    physics = ref.object_assets()[0].articulations[0].physics
    assert (articulation.q_id, articulation.dof_id, articulation.reference_id) == (7, 6, 0)
    assert articulation.drive == ref.object_assets()[0].articulations[0].drive
    assert builder.joint_armature[articulation.dof_id] == pytest.approx(physics.armature)
    assert builder.joint_friction[articulation.dof_id] == pytest.approx(physics.friction)
    np.testing.assert_allclose(solver.mjw_model.dof_armature.numpy()[:, articulation.dof_id], physics.armature)
    np.testing.assert_allclose(solver.mjw_model.dof_frictionloss.numpy()[:, articulation.dof_id], physics.friction)
    assert builder.joint_target_ke[articulation.dof_id] == 50.0
    assert builder.joint_target_kd[articulation.dof_id] == 2.0
    assert builder.joint_target_mode[articulation.dof_id] == int(newton.JointTargetMode.POSITION_VELOCITY)
    assert builder.joint_effort_limit[articulation.dof_id] == 50.0
    assert builder.joint_q[articulation.q_id] == pytest.approx(ref.object_articulation()[0, 0])
    reference_pos = ref.object_body_pos_w()[0]
    reference_quat_wxyz = ref.object_body_quat_w()[0]
    for body in binding.bodies:
        actual = body_q[body.body_id]
        np.testing.assert_allclose(actual[:3], reference_pos[body.reference_body_id], atol=1.0e-4)
        expected_xyzw = reference_quat_wxyz[body.reference_body_id][[1, 2, 3, 0]]
        actual_quat = actual[3:] / np.linalg.norm(actual[3:])
        expected_quat = expected_xyzw / np.linalg.norm(expected_xyzw)
        orientation_error = 2.0 * np.arccos(np.clip(abs(np.dot(actual_quat, expected_quat)), 0.0, 1.0))
        assert orientation_error < 1.0e-3


def test_object_import_does_not_collapse_or_remap_existing_builder_prefix():
    import warp as wp

    import newton
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.scene.objects import spawn_objects

    ref = load_mano_sharpa(str(_HOT3D))
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        root = builder.add_link(label="robot/root", mass=1.0)
        mount = builder.add_link(label="robot/preserved_mount", mass=0.1)
        builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z, label="robot/root_joint")
        builder.add_joint_fixed(parent=root, child=mount, label="robot/preserved_mount_joint")
        body_prefix = tuple(builder.body_label)
        joint_prefix = tuple(builder.joint_label)
        q_start_prefix = tuple(builder.joint_q_start)
        qd_start_prefix = tuple(builder.joint_qd_start)
        q_offset = len(builder.joint_q)

        bindings = spawn_objects(builder, ref, decompose_collision=False)

    assert tuple(builder.body_label[: len(body_prefix)]) == body_prefix
    assert tuple(builder.joint_label[: len(joint_prefix)]) == joint_prefix
    assert tuple(builder.joint_q_start[: len(q_start_prefix)]) == q_start_prefix
    assert tuple(builder.joint_qd_start[: len(qd_start_prefix)]) == qd_start_prefix
    assert [binding.root.body_id for binding in bindings] == [len(body_prefix), len(body_prefix) + 1]
    assert [binding.root.free_q_ids[0] for binding in bindings] == [q_offset, q_offset + 7]


def test_two_object_imports_preserve_vega_palm_mounts_and_layout():
    import warp as wp

    import newton
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.vega_sharpa import VegaSharpa
    from flash_chord.scene.objects import spawn_objects

    ref = load_mano_sharpa(str(_HOT3D))
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        layout = VegaSharpa().build(builder)
        body_prefix = tuple(builder.body_label)
        joint_prefix = tuple(builder.joint_label)

        bindings = spawn_objects(builder, ref, decompose_collision=False)

    assert len(body_prefix) == 60
    assert tuple(builder.body_label[: len(body_prefix)]) == body_prefix
    assert tuple(builder.joint_label[: len(joint_prefix)]) == joint_prefix
    for side in ("left", "right"):
        palm = layout.hand(side).palm_frame
        assert palm is not None
        assert builder.body_label[palm.body_id].endswith(f"/{side}_hand_C_MC")
    assert [binding.root.body_id for binding in bindings] == [60, 61]


def test_object_import_rejects_incomplete_reference_body_ownership_before_mutating_builder():
    import newton
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.scene.objects import spawn_objects

    reference = load_mano_sharpa(str(_HOT3D))
    incomplete = _with_object_assets(reference, reference.object_assets()[:1])
    builder = newton.ModelBuilder()

    with pytest.raises(ValueError, match="cover every reference body exactly once"):
        spawn_objects(builder, incomplete, decompose_collision=False)

    assert builder.body_count == 0
    assert builder.shape_count == 0


def test_object_import_rejects_missing_component_body_name():
    import newton
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.scene.objects import spawn_objects

    reference = load_mano_sharpa(str(_ARCTIC_BOX))
    asset = reference.object_assets()[0]
    bodies = tuple(
        replace(body, simulation_name="missing_top") if body.reference_name == "top" else body for body in asset.bodies
    )
    invalid = _with_object_assets(reference, (replace(asset, bodies=bodies),))

    with pytest.raises(ValueError, match="unable to resolve object bodies"):
        spawn_objects(newton.ModelBuilder(), invalid, decompose_collision=False)


def test_object_import_rejects_declared_root_without_free_joint():
    import newton
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.scene.objects import spawn_objects

    reference = load_mano_sharpa(str(_ARCTIC_BOX))
    asset = replace(reference.object_assets()[0], root_reference_name="top")
    invalid = _with_object_assets(reference, (asset,))

    with pytest.raises(ValueError, match="exactly one world-parented free joint"):
        spawn_objects(newton.ModelBuilder(), invalid, decompose_collision=False)
