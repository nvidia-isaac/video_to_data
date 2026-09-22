# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actor-only FlashSAC evaluation and checkpoint tests."""

import numpy as np
import pytest


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("safetensors")
    return jax, jnp


def _config(world_count=1):
    from flash_chord.training.flash_sac.config import NetworkConfig, ReplayConfig, TrainingConfig

    return TrainingConfig(
        world_count=world_count,
        total_environment_steps=10 * world_count,
        updates_per_collection=1.0,
        mixed_precision=False,
        replay=ReplayConfig(capacity=8, minimum_size=2, batch_size=2, n_step=1),
        network=NetworkConfig(
            actor_blocks=1,
            actor_hidden_dim=8,
            critic_blocks=1,
            critic_hidden_dim=8,
            expansion=2,
            atom_count=11,
        ),
    )


def test_evaluation_config_requires_a_checkpoint():
    _dependencies()

    from flash_chord.training.flash_sac.evaluation import EvaluationConfig

    assert EvaluationConfig("policy.safetensors").deterministic is True
    with pytest.raises(ValueError, match="must be specified"):
        EvaluationConfig("")


def test_evaluation_config_validates_reset_and_motion_overrides():
    _dependencies()

    from flash_chord.training.flash_sac.evaluation import EvaluationConfig

    assert EvaluationConfig("policy.safetensors", reset_mode="sampled_settled").reset_mode == "sampled_settled"
    with pytest.raises(ValueError, match="reset_mode"):
        EvaluationConfig("policy.safetensors", reset_mode="sampled")
    with pytest.raises(ValueError, match="motion_start_frame"):
        EvaluationConfig("policy.safetensors", motion_start_frame=-1)
    with pytest.raises(ValueError, match="motion_end_frame"):
        EvaluationConfig("policy.safetensors", motion_end_frame=-2)


@pytest.mark.parametrize("action_scale", [1.0, 10.0])
def test_compiled_deterministic_policy_matches_actor_mean_and_uses_batch_stats(action_scale):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.evaluation import create_actor_template, compile_policy_action
    from flash_chord.training.flash_sac.network import deterministic_action

    actor, state = create_actor_template(_config(), observation_dim=3, action_dim=2)
    observation = jnp.asarray([[0.2, -0.4, 0.7]], dtype=jnp.float32)
    key = jax.random.PRNGKey(4)
    compiled = compile_policy_action(actor, state, observation, key, deterministic=True, action_scale=action_scale)
    changed_state = state.replace(
        batch_stats=jax.tree.map(lambda value: value + 0.25, state.batch_stats),
    )
    actions = []
    for current_state in (state, changed_state):
        action, next_key = compiled(current_state, observation, key)
        expected = deterministic_action(
            actor.apply(
                {"params": current_state.params, "batch_stats": current_state.batch_stats},
                observation,
                training=False,
            ),
            action_scale=action_scale,
        )

        # XLA fusion can change float32 rounding relative to eager evaluation.
        # Apply the tolerance in normalized action units, independent of scaling.
        normalized = np.asarray(action) / action_scale
        np.testing.assert_allclose(normalized, np.asarray(expected) / action_scale, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_array_equal(np.asarray(next_key), np.asarray(key))
        assert action.shape == (1, 2)
        assert np.isfinite(np.asarray(action)).all()
        assert np.max(np.abs(np.asarray(action))) <= action_scale
        actions.append(normalized)

    # The same compiled executable must consume the supplied running statistics.
    assert not np.allclose(actions[0], actions[1], rtol=1.0e-3, atol=1.0e-3)


def test_policy_checkpoint_loads_for_a_different_evaluation_world_count(tmp_path):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.checkpoint import actor_inference_state, save_learner_checkpoint
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.learner import create_learner

    training = _config(world_count=2)
    learner, _ = create_learner(
        training,
        jnp.zeros((2, 3), dtype=jnp.float32),
        jnp.zeros((2, 3), dtype=jnp.float32),
        action_dim=2,
        objective_term_count=2,
    )
    paths = save_learner_checkpoint(tmp_path, learner)
    observation = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    state, policy_action, key, metadata = load_policy_for_inference(
        str(paths.policy),
        _config(world_count=1),
        observation,
        action_dim=2,
        deterministic=True,
    )
    action, _ = policy_action(state, observation, key)

    for restored, expected in zip(
        jax.tree.leaves(state),
        jax.tree.leaves(actor_inference_state(learner)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(expected))
    assert action.shape == (1, 2)
    assert metadata["algorithm"] == "flash_sac"
    assert metadata["checkpoint_kind"] == "policy"


def test_compiled_stochastic_policy_advances_key_and_remains_bounded():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.evaluation import create_actor_template, compile_policy_action

    actor, state = create_actor_template(_config(), observation_dim=3, action_dim=2)
    observation = jnp.zeros((1, 3), dtype=jnp.float32)
    key = jax.random.PRNGKey(9)
    compiled = compile_policy_action(actor, state, observation, key, deterministic=False)
    first, next_key = compiled(state, observation, key)
    second, final_key = compiled(state, observation, next_key)

    assert not np.array_equal(np.asarray(key), np.asarray(next_key))
    assert not np.array_equal(np.asarray(next_key), np.asarray(final_key))
    assert not np.array_equal(np.asarray(first), np.asarray(second))
    assert np.max(np.abs(np.asarray(first))) <= 1.0
    assert np.max(np.abs(np.asarray(second))) <= 1.0
