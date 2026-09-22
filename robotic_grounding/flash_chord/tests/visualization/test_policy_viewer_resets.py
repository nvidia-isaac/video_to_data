# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise repeated episode boundaries through both actual policy viewer step methods."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from flash_chord.evaluation.view import PolicyViewConfig
from flash_chord.runtime.jax import to_jax


@pytest.fixture(scope="module")
def policy_viewer_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "view_policy.py"
    spec = importlib.util.spec_from_file_location("policy_viewer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _EpisodeEnv:
    """Small CPU environment publishing the same transition buffers as WarpRLEnv."""

    def __init__(self, start_frame, world_count=1, auto_reset=False):
        self.world_count = world_count
        self.observation_dim = 1
        self.frame_dt = 0.02
        self.device = "cpu"
        self.auto_reset = auto_reset
        self.timestep = wp.full(world_count, start_frame, dtype=wp.int32, device="cpu")
        self.done = wp.zeros(world_count, dtype=wp.int32, device="cpu")
        self.truncation = wp.zeros(world_count, dtype=wp.int32, device="cpu")
        self.reset_mask = wp.zeros(world_count, dtype=wp.int32, device="cpu")
        self.observation = wp.full(world_count, float(start_frame), dtype=wp.float32, device="cpu")
        self.state_0 = object()
        self.next_done = np.zeros(world_count, dtype=np.int32)
        self.next_truncation = np.zeros(world_count, dtype=np.int32)
        self.resets = []

    def step(self, _action):
        frame = self.timestep.numpy() + 1
        self.done.assign(self.next_done)
        self.truncation.assign(self.next_truncation)
        self.reset_mask.zero_()
        if self.auto_reset:
            selected = np.maximum(self.next_done, self.next_truncation)
            frame[selected != 0] = 50  # Represent a sampled training recovery start.
            self.reset_mask.assign(selected)
        self.timestep.assign(frame)
        self.observation.assign(frame.astype(np.float32))
        return SimpleNamespace(observation=to_jax(self.observation, (self.world_count, 1)))

    def reset(self, *, frame_id, reset_mask):
        selected = reset_mask.numpy().astype(bool)
        self.resets.append((frame_id, selected.copy()))
        frame = self.timestep.numpy()
        frame[selected] = frame_id
        self.timestep.assign(frame)
        for buffer in (self.done, self.truncation, self.reset_mask):
            values = buffer.numpy()
            values[selected] = 0
            buffer.assign(values)
        self.observation.assign(frame.astype(np.float32))
        self.state_0 = object()
        return self.observation


def _viewer(module, monkeypatch, algorithm, env, evaluation):
    viewer_type = module.PPOPolicyViewer if algorithm == "ppo" else module.FlashSACPolicyViewer
    viewer = viewer_type.__new__(viewer_type)
    viewer.env = env
    viewer.jax_env = env
    viewer.evaluation = evaluation
    viewer.observation = jnp.array(to_jax(env.observation, (env.world_count, 1)), copy=True)
    viewer.state = env.state_0
    viewer.sim_time = 0.0
    viewer._next_action_frame = evaluation.start_frame
    viewer._shown_frame = evaluation.start_frame
    viewer._frame_status = wp.zeros(2, dtype=wp.int32, device="cpu")
    viewer.key = jax.random.key(42)
    inputs = []

    def action(observation):
        inputs.append(np.asarray(observation).copy())
        return jnp.zeros((env.world_count, 1))

    if algorithm == "ppo":
        viewer.learner = viewer.training = viewer.policy_evaluation = None
        monkeypatch.setattr(module, "ppo_policy_action", lambda _l, obs, _t, _e, _k: action(obs))
    else:
        viewer.actor_state = None
        viewer.policy_action = lambda _s, obs, key: (action(obs), key)
    return viewer, inputs


@pytest.mark.parametrize("algorithm", ["ppo", "flash_sac"])
@pytest.mark.parametrize("start_frame", [0, 17])
def test_explicit_mode_restarts_every_episode_and_uses_reset_observation(
    policy_viewer_module, monkeypatch, algorithm, start_frame
):
    with jax.default_device(jax.devices("cpu")[0]):
        env = _EpisodeEnv(start_frame)
        evaluation = PolicyViewConfig(checkpoint="unused", start_frame=start_frame)
        viewer, inputs = _viewer(policy_viewer_module, monkeypatch, algorithm, env, evaluation)

        for episode in range(12):
            env.next_done[:] = 0
            env.next_truncation[:] = 0
            viewer.step()
            assert inputs[-1][0, 0] == start_frame
            assert viewer._shown_frame == start_frame
            assert viewer.observation[0, 0] == start_frame + 1

            # Alternate failure and reference-end completion across repeated loops.
            env.next_done[:] = episode % 2
            env.next_truncation[:] = 1 - episode % 2
            viewer.step()
            assert inputs[-1][0, 0] == start_frame + 1
            assert viewer._shown_frame == start_frame
            assert viewer.observation[0, 0] == start_frame
            assert viewer.state is env.state_0
            if algorithm == "flash_sac":
                assert viewer._next_action_frame == start_frame

        assert [frame for frame, _ in env.resets] == [start_frame] * 12
        assert viewer.sim_time == pytest.approx(24 * env.frame_dt)


@pytest.mark.parametrize("algorithm", ["ppo", "flash_sac"])
@pytest.mark.parametrize("completed_world", [0, 1])
def test_explicit_mode_resets_only_the_completed_world(policy_viewer_module, monkeypatch, algorithm, completed_world):
    with jax.default_device(jax.devices("cpu")[0]):
        env = _EpisodeEnv(start_frame=7, world_count=2)
        evaluation = PolicyViewConfig(checkpoint="unused", start_frame=7)
        viewer, _ = _viewer(policy_viewer_module, monkeypatch, algorithm, env, evaluation)
        viewer.step()
        env.next_done[completed_world] = 1
        viewer.step()

        expected = [9, 9]
        expected[completed_world] = 7
        np.testing.assert_array_equal(env.timestep.numpy(), expected)
        np.testing.assert_array_equal(np.asarray(viewer.observation)[:, 0], expected)
        assert viewer._shown_frame == (7 if completed_world == 0 else 8)
        assert len(env.resets) == 1
        np.testing.assert_array_equal(env.resets[0][1], [completed_world == 0, completed_world == 1])


@pytest.mark.parametrize("algorithm", ["ppo", "flash_sac"])
def test_sampled_mode_preserves_environment_auto_reset(policy_viewer_module, monkeypatch, algorithm):
    with jax.default_device(jax.devices("cpu")[0]):
        env = _EpisodeEnv(start_frame=0, auto_reset=True)
        evaluation = PolicyViewConfig(checkpoint="unused", reset_mode="sampled_settled")
        viewer, inputs = _viewer(policy_viewer_module, monkeypatch, algorithm, env, evaluation)
        env.next_truncation[:] = 1
        viewer.step()
        assert viewer._shown_frame == 50
        assert viewer.observation[0, 0] == 50
        env.next_truncation[:] = 0
        viewer.step()
        assert inputs[-1][0, 0] == 50
        assert env.resets == []
