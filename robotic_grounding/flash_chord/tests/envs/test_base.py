# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the shared environment transition."""

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


def test_finish_transition_distinguishes_termination_truncation_and_active_worlds():
    import warp as wp

    from flash_chord.envs.base import finish_transition

    with wp.ScopedDevice("cuda:0"):
        timestep = wp.array([2, 4, 5, 2], dtype=wp.int32)
        episode_step = wp.array([10, 20, 30, 40], dtype=wp.int32)
        terminated = wp.array([1, 1, 0, 0], dtype=wp.int32)
        truncated = wp.array([0, 0, 1, 0], dtype=wp.int32)
        wp.launch(
            finish_transition,
            dim=4,
            inputs=[
                terminated,
                truncated,
            ],
            outputs=[timestep, episode_step],
        )

    assert terminated.numpy().tolist() == [1, 1, 0, 0]
    assert truncated.numpy().tolist() == [0, 0, 1, 0]
    assert timestep.numpy().tolist() == [2, 4, 5, 3]
    assert episode_step.numpy().tolist() == [10, 20, 30, 41]


@pytest.mark.sequence_data
def test_base_env_eager_and_captured_transition_agree():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.envs.base import BaseEnv, BaseEnvConfig
    from flash_chord.lifecycle.termination import Termination, TrackingTermination
    from flash_chord.objectives.composition import Objective
    from flash_chord.objectives.config import (
        ContactSupportObjectiveTermConfig,
        ObjectiveConfig,
        ObjectiveTermConfig,
        ShapedObjectiveTermConfig,
    )
    from flash_chord.runtime.actions import Action, ActionConfig, ResidualHandPoseAction
    from flash_chord.runtime.delay import TargetDelayConfig
    from flash_chord.runtime.sim import SimConfig
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy

    reference = load_mano_sharpa(str(_HOT3D), control_fps=50.0)
    embodiment = SharpaHands()
    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(
            embodiment,
            reference,
            world_count=1,
            collision=CollisionPolicy(robot_scope="none"),
            decompose_objects=False,
        )
        action_strategy = ResidualHandPoseAction.build(
            scene,
            config=ActionConfig(delay=TargetDelayConfig(min_steps=2, max_steps=2)),
        )
        objective_config = ObjectiveConfig(
            object_keypoints=ShapedObjectiveTermConfig(weight=0.0, enabled=False),
            hand_keypoints=ShapedObjectiveTermConfig(weight=0.0, enabled=False),
            hand_joint_pos=ShapedObjectiveTermConfig(weight=0.0, enabled=False),
            contact_wrench_support=ContactSupportObjectiveTermConfig(weight=0.0, enabled=False),
            missed_contact=ObjectiveTermConfig(weight=0.0, enabled=False),
            unintended_contact=ObjectiveTermConfig(weight=0.0, enabled=False),
            termination=ObjectiveTermConfig(weight=0.0),
            action_rate_l2=ObjectiveTermConfig(weight=0.0),
            action_l2=ObjectiveTermConfig(weight=0.0),
            num_wrench_basis=4,
            num_friction_cone_edges=4,
        )
        env = BaseEnv(
            scene,
            reference,
            config=BaseEnvConfig(sim=SimConfig(fps=50.0, substeps=5), objective=objective_config),
            action=action_strategy,
        )
        action = wp.zeros(env.action.action_dim, dtype=wp.float32)

        assert env.action is action_strategy
        assert isinstance(env.action, Action)
        assert isinstance(env.termination, Termination)
        assert isinstance(env.termination, TrackingTermination)
        assert isinstance(env.objective, Objective)
        assert env.contact is not None
        expected_contact_slots = sum(
            env.command.layout.num_bodies * scene.layout.hand(side).contact_link_count
            for side in env.contact.layout.sides
        )
        assert env.contact.layout.slots_per_world == expected_contact_slots

        env.reset(frame_id=0)
        env.step(action)
        eager_joint_q = env.state_0.joint_q.numpy().copy()
        eager_joint_qd = env.state_0.joint_qd.numpy().copy()
        assert env.timestep.numpy().tolist() == [1]
        assert env.episode_step.numpy().tolist() == [1]
        assert env.terminated.numpy().tolist() == [0]
        assert env.truncated.numpy().tolist() == [0]
        assert np.isfinite(eager_joint_q).all()
        assert np.isfinite(eager_joint_qd).all()
        assert np.isfinite(env.contact.contact_pos_b.numpy()).all()
        assert np.isfinite(env.contact.contact_force_direction_b.numpy()).all()
        np.testing.assert_allclose(env.objective.terms.numpy(), 0.0)
        np.testing.assert_allclose(env.objective.score.numpy(), 0.0)

        env.reset(frame_id=0)
        env.capture_transition()
        env.reset(frame_id=0)
        env.step(action)
        captured_joint_q = env.state_0.joint_q.numpy()
        captured_joint_qd = env.state_0.joint_qd.numpy()

    np.testing.assert_allclose(captured_joint_q, eager_joint_q, atol=1.0e-6)
    np.testing.assert_allclose(captured_joint_qd, eager_joint_qd, atol=1.0e-6)
