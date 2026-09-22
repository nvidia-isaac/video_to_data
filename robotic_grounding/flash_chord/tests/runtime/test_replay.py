# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-replay integration, two tiers:

* kinematic (collision-free + stiff tracking): the robot + objects track the reference to sub-cm,
  verifying the data -> reset -> control -> target pipeline end to end.
* physical (collision-on + default gains): grasp contact is active; the run stays stable (no NaN /
  blow-up) with bounded penetration. Contact fights the soft tracking, so drift is NOT required to be
  zero here (that offset is what an RL/MPPI policy later learns to correct).
* articulated physical: the production Dexmate composition tracks an interior contact-rich segment
  with zero residual, delayed arm targets, and solver-exported hand-object contact force.
"""

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.sequence_data]

_HOT3D = (
    ASSETS_DIR
    / "human_motion_data"
    / "hot3d"
    / "hot3d_processed"
    / "sequence_id=P0002_59a84a3a_seg025"
    / "robot_name=sharpa_wave"
)
_VEGA_MIXER = (
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s01_mixer_use_01"
    / "robot_name=vega_sharpa"
    / "data.parquet"
)


def test_kinematic_replay_tracks_reference_to_sub_cm():
    import warp as wp

    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands, SharpaHandsConfig
    from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy

    composed = compose_config(
        "replay",
        [
            "replay=kinematic",
            "action=residual_hand_pose_stiff",
            "collision=kinematic",
        ],
    )
    cfg = instantiate_typed(composed.replay, ReplayConfig)
    collision = instantiate_typed(composed.collision, CollisionPolicy)
    ref = load_mano_sharpa(str(_HOT3D), control_fps=cfg.sim.fps)
    embodiment = SharpaHands(SharpaHandsConfig.stiff_tracking())
    n_frames = min(40, ref.num_frames)

    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(
            embodiment,
            ref,
            world_count=1,
            collision=collision,
        )
        runner = ReplayRunner(scene, ref, cfg)
        runner.reset(0)
        peak_wrist = peak_obj = 0.0
        for f in range(n_frames):
            runner.step(f)
            assert not np.isnan(runner.state_0.joint_q.numpy()).any()
            res = runner.check_termination()
            assert not res.any, f"frame {f}: wrist={res.max_wrist_pos:.3f} obj={res.max_object_pos:.3f}"
            peak_wrist = max(peak_wrist, res.max_wrist_pos)
            peak_obj = max(peak_obj, res.max_object_pos)

    assert peak_wrist < 0.01, f"kinematic wrist tracking {peak_wrist*1000:.1f} mm exceeds 10 mm"
    assert peak_obj < 0.01, f"kinematic object tracking {peak_obj*1000:.1f} mm exceeds 10 mm"


def test_physical_replay_is_stable_with_bounded_penetration():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.runtime.contact import contact_gaps_w, read_contacts_w
    from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
    from flash_chord.scene.builder import build_scene

    cfg = ReplayConfig()
    ref = load_mano_sharpa(str(_HOT3D), control_fps=cfg.sim.fps)
    embodiment = SharpaHands()
    n_frames = min(40, ref.num_frames)

    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(embodiment, ref, world_count=1)  # collision on
        runner = ReplayRunner(scene, ref, cfg)
        runner.reset(0)
        worst_pen = 0.0
        peak_contact_force = 0.0
        for f in range(n_frames):
            runner.step(f)
            assert not np.isnan(runner.state_0.joint_q.numpy()).any(), f"NaN at frame {f}"
            res = runner.check_termination()
            assert res.max_object_pos < 0.5 and res.max_wrist_pos < 0.5, f"blow-up at frame {f}"
            gaps = contact_gaps_w(scene.model, runner.state_0, runner.contacts)
            if gaps.size:
                worst_pen = min(worst_pen, float(gaps.min()))
            readout = read_contacts_w(scene.model, runner.state_0, runner.contacts)
            if len(readout):
                peak_contact_force = max(
                    peak_contact_force,
                    float(np.linalg.norm(readout.contact_force_w, axis=1).max()),
                )

    assert worst_pen > -0.01, f"penetration {worst_pen*1000:.1f} mm exceeds 10 mm"
    assert peak_contact_force > 0.0


def test_vega_physical_replay_tracks_with_zero_residual_and_exports_contact_force():
    import warp as wp

    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    composed = compose_config(
        "replay",
        [
            f"task.parquet='{_VEGA_MIXER}'",
            "task.motion_speed=0.5",
            "sim=rl",
            "embodiment=dexmate_sharpa",
            "action=residual_joint_position",
            "collision=manipulation",
            "collision.robot_support_collision=true",
        ],
    )
    replay = instantiate_typed(composed.replay, ReplayConfig)
    embodiment = instantiate_typed(composed.embodiment, Embodiment)
    collision = instantiate_typed(composed.collision, CollisionPolicy)
    start_frame = 100

    with wp.ScopedDevice("cuda:0"):
        setup = setup_scene(
            parquet=composed.task.parquet,
            control_fps=replay.sim.fps,
            motion_speed=composed.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=composed.scene.world_count,
            include_support=composed.scene.support,
            decompose_objects=composed.scene.decompose_objects,
        )
        scene = setup.scene
        reference = setup.reference
        runner = ReplayRunner(scene, reference, replay)
        runner.reset(start_frame)

        np.testing.assert_allclose(
            runner.state_0.joint_q.numpy()[: scene.layout.num_joint_q],
            scene.robot_reference.joint_q[start_frame],
            atol=1.0e-6,
        )

        observed_hand_object_force = False
        hand_shapes = set(scene.collision_layout.contact_shape_ids)
        object_shapes = set(scene.collision_layout.objects.ids())
        for frame in range(start_frame, start_frame + 4):
            runner.step(frame)
            np.testing.assert_allclose(
                runner.action.joint_target.numpy()[: scene.layout.num_joint_dof],
                scene.robot_reference.joint_target[frame],
                atol=1.0e-6,
            )
            np.testing.assert_allclose(runner.action.action_l2.numpy(), 0.0)
            np.testing.assert_allclose(runner.action.action_rate_l2.numpy(), 0.0)
            for value in (
                runner.state_0.joint_q,
                runner.state_0.joint_qd,
                runner.state_0.body_q,
                runner.state_0.body_qd,
                runner.action.joint_target,
            ):
                assert np.isfinite(value.numpy()).all()

            termination = runner.check_termination()
            assert not termination.any
            assert termination.max_wrist_pos < 0.1
            assert termination.max_object_pos < 0.1

            count = int(runner.contacts.rigid_contact_count.numpy()[0])
            shape0 = runner.contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = runner.contacts.rigid_contact_shape1.numpy()[:count]
            force = runner.contacts.force.numpy()[:count, :3]
            for contact, (first, second) in enumerate(zip(shape0, shape1, strict=True)):
                is_hand_object = (
                    int(first) in hand_shapes and int(second) in object_shapes
                ) or (
                    int(second) in hand_shapes and int(first) in object_shapes
                )
                observed_hand_object_force |= is_hand_object and np.linalg.norm(force[contact]) > 0.0

        assert observed_hand_object_force
