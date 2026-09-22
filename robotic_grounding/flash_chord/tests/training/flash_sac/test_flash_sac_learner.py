# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end JIT tests for the native FlashSAC learner state transition."""

import numpy as np
import pytest


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    return jax, jnp


def _config(
    *,
    total_environment_steps=20,
    updates_per_collection=2.0,
    replay=None,
    mixed_precision=False,
    reward_normalization=True,
    optimization=None,
    network=None,
    exploration=None,
):
    from flash_chord.training.flash_sac.config import (
        ExplorationConfig,
        NetworkConfig,
        OptimizationConfig,
        ReplayConfig,
        RewardNormalizationConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        world_count=2,
        total_environment_steps=total_environment_steps,
        updates_per_collection=updates_per_collection,
        mixed_precision=mixed_precision,
        compute_dtype="float16" if mixed_precision else "float32",
        replay=replay or ReplayConfig(capacity=8, minimum_size=2, batch_size=2, n_step=1),
        network=network
        or NetworkConfig(
            actor_blocks=1,
            actor_hidden_dim=8,
            critic_blocks=1,
            critic_hidden_dim=8,
            expansion=2,
            atom_count=11,
        ),
        optimization=optimization or OptimizationConfig(),
        exploration=exploration or ExplorationConfig(),
        reward_normalization=RewardNormalizationConfig(enabled=reward_normalization),
    )


def _initial(jnp):
    observation = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10.0
    context = jnp.asarray([[1.0, 0.5, 0.0], [0.75, 0.25, 1.0]], dtype=jnp.float32)
    return observation, context


def _create(config, jnp, *, termination_cause_count=1, tracking_error_count=1):
    from flash_chord.training.flash_sac.learner import create_learner

    observation, context = _initial(jnp)
    return create_learner(
        config,
        observation,
        context,
        action_dim=2,
        objective_term_count=2,
        termination_cause_count=termination_cause_count,
        tracking_error_count=tracking_error_count,
    )


def _transition(jnp, action, *, step=0, invalid_action=False):
    from flash_chord.training.flash_sac.learner import Transition

    observation, context = _initial(jnp)
    if invalid_action:
        action = action.at[0, 0].set(jnp.nan)
    terminated = jnp.asarray([False, step % 3 == 2], dtype=jnp.int32)
    truncated = jnp.asarray([step % 2 == 1, False], dtype=jnp.int32)
    completed = jnp.logical_or(terminated, truncated)
    return Transition(
        action=action,
        objective_terms=jnp.asarray([[1.0 + step, -0.5], [0.25, 0.75 + step]], dtype=jnp.float32),
        terminated=terminated,
        truncated=truncated,
        episode_reference_progress=completed.astype(jnp.float32) * 0.5,
        termination_diagnostics=jnp.stack(
            (
                jnp.asarray([0.0, 1.0 + step], dtype=jnp.float32),
                jnp.asarray([terminated[1], 2.0 + step], dtype=jnp.float32),
            )
        ),
        next_observation=observation + 1.0 + step,
        next_critic_context=context + 0.25 + step,
        post_reset_observation=observation + 2.0 + step,
        post_reset_critic_context=context + 0.5 + step,
    )


def _first_action(state, executables):
    action, exploration, key = executables.initial_action(
        state.actor.params,
        state.actor.batch_stats,
        state.replay.size,
        state.observation,
        state.exploration,
        state.key,
    )
    return state.replace(exploration=exploration, key=key), action


def _consume(
    state,
    executables,
    transition,
    jnp,
    *,
    weights=(1.0, 0.5),
    target_voc_scale=0.25,
    stage_changed=False,
    reset_logging=False,
):
    return executables.consume_transition(
        state,
        transition,
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(0.05, dtype=jnp.float32),
        jnp.asarray(target_voc_scale, dtype=jnp.float32),
        jnp.asarray(stage_changed, dtype=jnp.bool_),
        jnp.asarray(reset_logging, dtype=jnp.bool_),
    )


def _tree_numpy(jax, tree):
    return [np.array(value, copy=True) for value in jax.tree.leaves(tree)]


def _assert_tree_equal(jax, actual, expected):
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_leaf), expected_leaf)


def _assert_tree_allclose(jax, actual, expected, *, atol=1.0e-7):
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), expected, strict=True):
        np.testing.assert_allclose(np.asarray(actual_leaf), expected_leaf, rtol=1.0e-6, atol=atol)


def test_create_learner_initializes_fixed_float32_state_and_independent_target():
    jax, jnp = _dependencies()

    state, executables = _create(_config(), jnp)

    assert state.observation.shape == (2, 3)
    assert state.critic_context.shape == (2, 3)
    assert state.replay.storage.observation.shape == (8, 3)
    assert state.replay.storage.action.shape == (8, 2)
    assert state.replay.storage.discounted_objective_terms.shape == (8, 2)
    assert state.loss_scale.scale == 1.0
    assert state.environment_steps.dtype == jnp.int32
    assert state.global_update_step.dtype == jnp.int32
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(state.actor.params))
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(state.critic.params))
    assert hasattr(executables.initial_action, "lower")
    assert hasattr(executables.consume_transition, "lower")

    critic_leaf = jax.tree.leaves(state.critic.params)[0]
    target_leaf = jax.tree.leaves(state.target_critic.params)[0]
    np.testing.assert_array_equal(np.asarray(target_leaf), np.asarray(critic_leaf))
    assert critic_leaf.unsafe_buffer_pointer() != target_leaf.unsafe_buffer_pointer()


def test_create_learner_rejects_incomplete_critic_context():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.learner import create_learner

    observation, _ = _initial(jnp)
    with pytest.raises(ValueError, match="applied_voc_scale"):
        create_learner(
            _config(),
            observation,
            jnp.zeros((2, 2), dtype=jnp.float32),
            action_dim=2,
            objective_term_count=2,
        )


def test_expanded_critic_context_preserves_markov_fields_across_stage_change():
    _, jnp = _dependencies()

    from flash_chord.lifecycle.reset import RESET_CONTEXT_NAMES
    from flash_chord.training.flash_sac.learner import create_learner

    observation, reset_context = _initial(jnp)
    extra_context = jnp.asarray(
        [[0.1, 1.0, 2.0], [0.2, 3.0, 4.0]],
        dtype=jnp.float32,
    )
    context = jnp.concatenate((reset_context, extra_context), axis=-1)
    context_names = RESET_CONTEXT_NAMES + (
        "normalized_reference_phase",
        "object_body_0_linear_velocity_w_x",
        "object_body_0_linear_velocity_w_y",
    )
    state, executables = create_learner(
        _config(updates_per_collection=0.5),
        observation,
        context,
        action_dim=2,
        objective_term_count=2,
        termination_cause_count=1,
        tracking_error_count=1,
        critic_context_names=context_names,
    )
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    transition = _transition(jnp, action).replace(
        next_critic_context=context + 0.25,
        post_reset_critic_context=context + 0.5,
    )

    state, _, _ = _consume(
        state,
        executables,
        transition,
        jnp,
        target_voc_scale=0.1,
        stage_changed=True,
    )

    np.testing.assert_allclose(np.asarray(state.replay.storage.critic_context[:2, 3:]), extra_context)
    np.testing.assert_allclose(np.asarray(state.critic_context[:, 3:]), np.asarray(extra_context + 0.5))


@pytest.mark.parametrize("observation_storage_dtype", ["float32", "float16"])
def test_one_donated_call_crosses_warmup_runs_scan_and_returns_next_action(observation_storage_dtype):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import ReplayConfig

    observation, context = _initial(jnp)
    replay = ReplayConfig(
        capacity=8,
        minimum_size=2,
        batch_size=2,
        n_step=1,
        observation_storage_dtype=observation_storage_dtype,
    )
    state, executables = _create(_config(replay=replay), jnp)
    state, action = _first_action(state, executables)
    assert action.shape == (2, 2)
    assert np.all(np.abs(np.asarray(action)) <= 1.0)
    initial_actor_key = np.asarray(state.key)

    state, next_action, metrics = _consume(state, executables, _transition(jnp, action), jnp)
    jax.block_until_ready(next_action)

    assert next_action.shape == (2, 2)
    assert np.all(np.abs(np.asarray(next_action)) <= 1.0)
    assert int(state.replay.size) == 2
    assert state.replay.storage.observation.dtype == jnp.dtype(observation_storage_dtype)
    assert int(state.environment_steps) == 2
    assert int(state.global_update_step) == 2
    assert int(state.actor.schedule_step) == 1
    assert int(state.temperature.schedule_step) == 1
    assert int(state.critic.schedule_step) == 2
    assert int(metrics.actor_update_count) == 1
    assert int(metrics.critic_update_count) == 2
    assert float(metrics.action_abs_mean_per_env_action) == pytest.approx(float(np.abs(np.asarray(action)).mean()))
    assert float(state.update_credit) == 0.0
    np.testing.assert_allclose(np.asarray(state.observation), np.asarray(observation + 2.0))
    np.testing.assert_allclose(np.asarray(state.critic_context), np.asarray(context + 0.5))
    assert not np.array_equal(np.asarray(state.key), initial_actor_key)
    for value in jax.tree.leaves(metrics):
        assert np.isfinite(np.asarray(value)).all()


def test_explicit_learning_rate_horizon_makes_common_update_prefix_budget_independent():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import OptimizationConfig

    optimization = OptimizationConfig(learning_rate_schedule_updates=8)
    short_state, short_executables = _create(
        _config(total_environment_steps=20, optimization=optimization),
        jnp,
    )
    long_state, long_executables = _create(
        _config(total_environment_steps=40, optimization=optimization),
        jnp,
    )
    short_state, short_action = _first_action(short_state, short_executables)
    long_state, long_action = _first_action(long_state, long_executables)

    for step in range(3):
        np.testing.assert_array_equal(np.asarray(short_action), np.asarray(long_action))
        transition = _transition(jnp, short_action, step=step)
        short_state, short_action, _ = _consume(short_state, short_executables, transition, jnp)
        long_state, long_action, _ = _consume(long_state, long_executables, transition, jnp)

    jax.block_until_ready((short_state, long_state))
    _assert_tree_equal(jax, short_state.actor, _tree_numpy(jax, long_state.actor))
    _assert_tree_equal(jax, short_state.critic, _tree_numpy(jax, long_state.critic))
    _assert_tree_equal(jax, short_state.temperature, _tree_numpy(jax, long_state.temperature))
    assert int(short_state.actor.schedule_step) == int(long_state.actor.schedule_step) == 3
    assert int(short_state.critic.schedule_step) == int(long_state.critic.schedule_step) == 6
    assert int(short_state.temperature.schedule_step) == int(long_state.temperature.schedule_step) == 3


def test_residual_mean_update_scaling_is_selective_and_default_is_exact_noop():
    jax, jnp = _dependencies()
    from flax.core import freeze

    from flash_chord.training.flash_sac.learner import _scale_residual_mean_updates

    updates = freeze(
        {
            "embed": {"unit_kernel": jnp.asarray([[1.0, 2.0]])},
            "log_std": {
                "gain": jnp.asarray([3.0, 4.0]),
                "unit_kernel": jnp.asarray([[5.0, 6.0]]),
            },
            "log_std_bias": jnp.asarray([7.0, 8.0]),
            "mean": {
                "gain": jnp.asarray([9.0, 10.0]),
                "unit_kernel": jnp.asarray([[11.0, 12.0]]),
            },
            "mean_bias": jnp.asarray([13.0, 14.0]),
        }
    )

    default = _scale_residual_mean_updates(updates, 1.0)
    scaled = _scale_residual_mean_updates(updates, 0.1)

    assert type(default) is type(updates)
    assert type(scaled) is type(updates)
    for actual, expected in zip(default.values(), updates.values(), strict=True):
        for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))
    np.testing.assert_allclose(np.asarray(scaled["mean"]["gain"]), [0.9, 1.0])
    np.testing.assert_allclose(np.asarray(scaled["mean_bias"]), [1.3, 1.4])
    np.testing.assert_array_equal(scaled["mean"]["unit_kernel"], updates["mean"]["unit_kernel"])
    for name in ("embed", "log_std", "log_std_bias"):
        for actual_leaf, expected_leaf in zip(
            jax.tree.leaves(scaled[name]),
            jax.tree.leaves(updates[name]),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))
    np.testing.assert_array_equal(np.asarray(updates["mean"]["gain"]), [9.0, 10.0])
    np.testing.assert_array_equal(np.asarray(updates["mean_bias"]), [13.0, 14.0])


def test_zero_dense_mean_update_scaling_is_selective():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.learner import _scale_residual_mean_updates

    updates = {
        "embed": {"unit_kernel": jnp.asarray([[1.0, 2.0]])},
        "log_std": {"gain": jnp.asarray([3.0, 4.0])},
        "mean": {"kernel": jnp.asarray([[5.0, 6.0], [7.0, 8.0]])},
        "mean_bias": jnp.asarray([9.0, 10.0]),
    }

    scaled = _scale_residual_mean_updates(updates, 0.1, mean_head="zero_dense")

    np.testing.assert_allclose(np.asarray(scaled["mean"]["kernel"]), [[0.5, 0.6], [0.7, 0.8]])
    np.testing.assert_allclose(np.asarray(scaled["mean_bias"]), [0.9, 1.0])
    for name in ("embed", "log_std"):
        _assert_tree_equal(jax, scaled[name], _tree_numpy(jax, updates[name]))
    np.testing.assert_array_equal(np.asarray(updates["mean"]["kernel"]), [[5.0, 6.0], [7.0, 8.0]])


@pytest.mark.parametrize(
    "updates",
    [
        {"mean": {"gain": np.ones(2, dtype=np.float32)}},
        {"mean": {"unit_kernel": np.ones((2, 2), dtype=np.float32)}, "mean_bias": np.ones(2, dtype=np.float32)},
        np.ones(2, dtype=np.float32),
    ],
)
def test_residual_mean_update_scaling_rejects_incompatible_parameter_paths(updates):
    _dependencies()

    from flash_chord.training.flash_sac.learner import _scale_residual_mean_updates

    with pytest.raises((TypeError, ValueError), match="residual actor updates|actor updates"):
        _scale_residual_mean_updates(updates, 0.1)


@pytest.mark.parametrize(
    ("residual_mean_head", "parameter_name"),
    [("unit_gain", "gain"), ("zero_dense", "kernel")],
)
def test_residual_mean_multiplier_scales_only_mean_head_and_bias_optimizer_steps(
    residual_mean_head,
    parameter_name,
):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import NetworkConfig, OptimizationConfig

    network = NetworkConfig(
        actor_blocks=1,
        actor_hidden_dim=8,
        actor_head_mode="residual",
        residual_mean_head=residual_mean_head,
        initial_normalized_std=0.1,
        critic_blocks=1,
        critic_hidden_dim=8,
        expansion=2,
        atom_count=11,
    )
    base_config = _config(
        updates_per_collection=1.0,
        network=network,
        optimization=OptimizationConfig(residual_mean_learning_rate_multiplier=1.0),
    )
    scaled_config = _config(
        updates_per_collection=1.0,
        network=network,
        optimization=OptimizationConfig(residual_mean_learning_rate_multiplier=0.25),
    )
    base_state, base_executables = _create(base_config, jnp)
    scaled_state, scaled_executables = _create(scaled_config, jnp)
    base_state, base_action = _first_action(base_state, base_executables)
    scaled_state, scaled_action = _first_action(scaled_state, scaled_executables)
    np.testing.assert_array_equal(np.asarray(scaled_action), np.asarray(base_action))

    initial_mean_parameter = np.array(base_state.actor.params["mean"][parameter_name], copy=True)
    initial_bias = np.array(base_state.actor.params["mean_bias"], copy=True)
    transition = _transition(jnp, base_action)
    base_state, _, base_metrics = _consume(base_state, base_executables, transition, jnp)
    scaled_state, _, scaled_metrics = _consume(scaled_state, scaled_executables, transition, jnp)
    jax.block_until_ready((base_state, scaled_state))

    base_mean_delta = np.asarray(base_state.actor.params["mean"][parameter_name]) - initial_mean_parameter
    scaled_mean_delta = np.asarray(scaled_state.actor.params["mean"][parameter_name]) - initial_mean_parameter
    base_bias_delta = np.asarray(base_state.actor.params["mean_bias"]) - initial_bias
    scaled_bias_delta = np.asarray(scaled_state.actor.params["mean_bias"]) - initial_bias
    assert np.any(base_mean_delta != 0.0)
    assert np.any(base_bias_delta != 0.0)
    np.testing.assert_allclose(scaled_mean_delta, 0.25 * base_mean_delta, rtol=1.0e-5, atol=1.0e-8)
    np.testing.assert_allclose(scaled_bias_delta, 0.25 * base_bias_delta, rtol=1.0e-5, atol=1.0e-8)

    for name in base_state.actor.params:
        if name == "mean_bias":
            continue
        if name == "mean":
            for leaf_name in base_state.actor.params[name]:
                if leaf_name != parameter_name:
                    np.testing.assert_array_equal(
                        np.asarray(scaled_state.actor.params[name][leaf_name]),
                        np.asarray(base_state.actor.params[name][leaf_name]),
                    )
            continue
        _assert_tree_equal(jax, scaled_state.actor.params[name], _tree_numpy(jax, base_state.actor.params[name]))
    _assert_tree_equal(jax, scaled_state.actor.optimizer_state, _tree_numpy(jax, base_state.actor.optimizer_state))
    _assert_tree_equal(jax, scaled_state.actor.batch_stats, _tree_numpy(jax, base_state.actor.batch_stats))
    _assert_tree_equal(jax, scaled_state.temperature, _tree_numpy(jax, base_state.temperature))
    assert int(scaled_state.actor.schedule_step) == int(base_state.actor.schedule_step) == 1
    assert float(scaled_metrics.actor_loss_mean_per_update) == float(base_metrics.actor_loss_mean_per_update)
    assert float(scaled_metrics.actor_entropy_mean_per_update) == float(base_metrics.actor_entropy_mean_per_update)


def test_actor_learning_rate_multiplier_scales_trunk_and_both_policy_heads():
    jax, jnp = _dependencies()

    from flax.core import unfreeze

    from flash_chord.training.flash_sac.config import NetworkConfig, OptimizationConfig

    network = NetworkConfig(
        actor_blocks=1,
        actor_hidden_dim=8,
        actor_head_mode="residual",
        initial_normalized_std=0.1,
        critic_blocks=1,
        critic_hidden_dim=8,
        expansion=2,
        atom_count=11,
    )
    base_config = _config(
        updates_per_collection=1.0,
        network=network,
        optimization=OptimizationConfig(actor_learning_rate_multiplier=1.0),
    )
    scaled_config = _config(
        updates_per_collection=1.0,
        network=network,
        optimization=OptimizationConfig(actor_learning_rate_multiplier=0.25),
    )
    base_state, base_executables = _create(base_config, jnp)
    scaled_state, scaled_executables = _create(scaled_config, jnp)

    def _with_nonzero_head_gains(state):
        params = unfreeze(state.actor.params)
        for head_name in ("mean", "log_std"):
            params[head_name]["gain"] = jnp.full_like(params[head_name]["gain"], 0.1)
        return state.replace(actor=state.actor.replace(params=params))

    base_state = _with_nonzero_head_gains(base_state)
    scaled_state = _with_nonzero_head_gains(scaled_state)
    base_state, base_action = _first_action(base_state, base_executables)
    scaled_state, scaled_action = _first_action(scaled_state, scaled_executables)
    np.testing.assert_array_equal(np.asarray(scaled_action), np.asarray(base_action))

    initial_params = jax.tree.map(lambda value: np.array(value, copy=True), base_state.actor.params)
    transition = _transition(jnp, base_action)
    base_state, _, base_metrics = _consume(base_state, base_executables, transition, jnp)
    scaled_state, _, scaled_metrics = _consume(scaled_state, scaled_executables, transition, jnp)
    jax.block_until_ready((base_state, scaled_state))

    for parameter_name in ("mean_bias", "log_std_bias"):
        initial = np.asarray(initial_params[parameter_name])
        base_delta = np.asarray(base_state.actor.params[parameter_name]) - initial
        scaled_delta = np.asarray(scaled_state.actor.params[parameter_name]) - initial
        assert np.any(base_delta != 0.0)
        np.testing.assert_allclose(scaled_delta, 0.25 * base_delta, rtol=3.0e-4, atol=2.0e-8)

    for module_name in ("embed", "mean", "log_std"):
        initial = np.asarray(initial_params[module_name]["unit_kernel"])
        base_delta = np.asarray(base_state.actor.params[module_name]["unit_kernel"]) - initial
        scaled_delta = np.asarray(scaled_state.actor.params[module_name]["unit_kernel"]) - initial
        assert np.linalg.norm(base_delta) > 0.0
        assert np.linalg.norm(scaled_delta) < np.linalg.norm(base_delta)
        assert not np.array_equal(
            np.asarray(scaled_state.actor.params[module_name]["unit_kernel"]),
            np.asarray(base_state.actor.params[module_name]["unit_kernel"]),
        )

    _assert_tree_equal(jax, scaled_state.actor.optimizer_state, _tree_numpy(jax, base_state.actor.optimizer_state))
    _assert_tree_equal(jax, scaled_state.actor.batch_stats, _tree_numpy(jax, base_state.actor.batch_stats))
    _assert_tree_equal(jax, scaled_state.temperature, _tree_numpy(jax, base_state.temperature))
    assert int(scaled_state.actor.schedule_step) == int(base_state.actor.schedule_step) == 1
    assert float(scaled_metrics.actor_loss_mean_per_update) == float(base_metrics.actor_loss_mean_per_update)
    assert float(scaled_metrics.actor_entropy_mean_per_update) == float(base_metrics.actor_entropy_mean_per_update)


def test_default_uniform_warmup_matches_upstream_action_before_replay_minimum():
    jax, jnp = _dependencies()

    state, executables = _create(_config(), jnp)
    expected_key, action_key = jax.random.split(state.key)
    expected_action = jax.random.uniform(
        action_key,
        (2, 2),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )

    state, action = _first_action(state, executables)

    np.testing.assert_array_equal(np.asarray(action), np.asarray(expected_action))
    np.testing.assert_array_equal(jax.random.key_data(state.key), jax.random.key_data(expected_key))
    assert int(state.exploration.repeat_count) == 0
    assert int(state.exploration.repeat_length) == 0


def test_policy_warmup_uses_residual_actor_before_replay_minimum():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import ExplorationConfig, NetworkConfig, ReplayConfig

    config = _config(
        replay=ReplayConfig(capacity=8, minimum_size=8, batch_size=2, n_step=1),
        network=NetworkConfig(
            actor_blocks=1,
            actor_hidden_dim=8,
            actor_head_mode="residual",
            initial_normalized_std=0.1,
            critic_blocks=1,
            critic_hidden_dim=8,
            expansion=2,
            atom_count=11,
        ),
        exploration=ExplorationConfig(warmup_action="policy"),
    )
    state, executables = _create(config, jnp)

    state, action = _first_action(state, executables)

    assert int(state.replay.size) == 0
    assert int(state.exploration.repeat_count) == 1
    assert int(state.exploration.repeat_length) >= 1
    assert np.max(np.abs(np.asarray(action))) < 1.0


def test_logging_window_accumulates_inside_donated_call_and_resets_after_emission():
    _, jnp = _dependencies()

    state, executables = _create(_config(), jnp)
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    for step in range(3):
        state, _, metrics = _consume(
            state,
            executables,
            _transition(jnp, action, step=step),
            jnp,
            reset_logging=step == 2,
        )

    assert int(metrics.actor_update_count) == 3
    assert int(metrics.critic_update_count) == 6
    assert float(metrics.reward_mean_per_env_step) == pytest.approx(0.071875)
    np.testing.assert_allclose(
        np.asarray(metrics.objective_contribution_mean_per_env_step),
        [0.05625, 0.015625],
        rtol=1.0e-6,
    )
    assert float(metrics.episode_return_mean_per_completed_episode) == pytest.approx(0.146875)
    assert float(metrics.episode_length_mean_per_completed_episode) == pytest.approx(2.5)
    assert float(metrics.episode_reference_progress_mean_per_completed_episode) == pytest.approx(0.5)
    assert int(metrics.episode_count) == 2
    assert int(metrics.reference_end_count) == 1
    np.testing.assert_array_equal(np.asarray(metrics.termination_cause_count), [1])
    np.testing.assert_allclose(np.asarray(metrics.tracking_error_mean_per_env_step), [2.5])

    assert int(state.logging.environment_step_count) == 0
    assert int(state.logging.episode_count) == 0
    np.testing.assert_allclose(np.asarray(state.logging.running_episode_return), [0.1375, 0.0])
    np.testing.assert_array_equal(np.asarray(state.logging.running_episode_length), [1, 0])


def test_logging_window_assigns_overlapping_termination_causes_exclusively_in_name_order():
    _, jnp = _dependencies()

    state, executables = _create(
        _config(),
        jnp,
        termination_cause_count=2,
        tracking_error_count=1,
    )
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    transition = _transition(jnp, action, step=2).replace(
        termination_diagnostics=jnp.asarray(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
            dtype=jnp.float32,
        )
    )
    _, _, metrics = _consume(state, executables, transition, jnp, reset_logging=True)

    assert int(metrics.episode_count) == 1
    np.testing.assert_array_equal(np.asarray(metrics.termination_cause_count), [1, 0])


def test_reward_statistics_include_current_transition_before_same_call_updates():
    _, jnp = _dependencies()

    state, executables = _create(_config(updates_per_collection=0.5), jnp)
    state, action = _first_action(state, executables)
    transition = _transition(jnp, action)
    state, _, metrics = _consume(state, executables, transition, jnp, weights=(2.0, -1.0))

    expected_reward = 0.05 * jnp.einsum(
        "wt,t->w",
        transition.objective_terms,
        jnp.asarray([2.0, -1.0]),
    )
    np.testing.assert_allclose(np.asarray(state.reward_normalizer.discounted_return), np.asarray(expected_reward))
    assert int(metrics.critic_update_count) == 0
    assert float(state.update_credit) == 0.5

    state, _, metrics = _consume(state, executables, _transition(jnp, action, step=1), jnp)
    assert int(metrics.critic_update_count) == 1
    assert int(state.global_update_step) == 1
    assert float(state.update_credit) == 0.0


def test_stage_change_flushes_pending_without_dropping_committed_replay():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import ReplayConfig

    config = _config(
        updates_per_collection=0.5,
        replay=ReplayConfig(capacity=8, minimum_size=8, batch_size=2, n_step=3),
    )
    state, executables = _create(config, jnp)
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    state, _, _ = _consume(state, executables, _transition(jnp, action), jnp)
    assert int(state.replay.pending.size) == 1
    assert int(state.replay.size) == 0

    state, _, _ = _consume(
        state,
        executables,
        _transition(jnp, action, step=1),
        jnp,
        target_voc_scale=0.1,
        stage_changed=True,
    )
    assert int(state.replay.pending.size) == 1
    assert int(state.replay.size) == 0
    np.testing.assert_allclose(np.asarray(state.replay.pending.critic_context[0, :, 1]), 0.1)
    np.testing.assert_allclose(np.asarray(state.replay.pending.critic_context[0, 1, 0]), 0.1)
    np.testing.assert_allclose(np.asarray(state.replay.pending.critic_context[0, 0, 0]), 1.5)


def test_configurable_stage_clear_retains_only_new_stage_entries():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import ReplayConfig

    replay = ReplayConfig(
        capacity=8,
        minimum_size=8,
        batch_size=2,
        n_step=1,
        clear_on_stage_change=True,
    )
    state, executables = _create(_config(updates_per_collection=0.5, replay=replay), jnp)
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    state, _, _ = _consume(state, executables, _transition(jnp, action), jnp)
    assert int(state.replay.size) == 2

    state, _, _ = _consume(
        state,
        executables,
        _transition(jnp, action, step=1),
        jnp,
        target_voc_scale=0.1,
        stage_changed=True,
    )
    assert int(state.replay.size) == 2


def test_policy_warmup_remains_policy_driven_after_stage_clear_below_minimum():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import ExplorationConfig, NetworkConfig, ReplayConfig
    from flash_chord.training.flash_sac.exploration import collection_action, truncated_zeta_cdf
    from flash_chord.training.flash_sac.network import Actor

    network = NetworkConfig(
        actor_blocks=1,
        actor_hidden_dim=8,
        actor_head_mode="residual",
        initial_normalized_std=0.1,
        critic_blocks=1,
        critic_hidden_dim=8,
        expansion=2,
        atom_count=11,
    )
    exploration = ExplorationConfig(warmup_action="policy")
    replay = ReplayConfig(
        capacity=8,
        minimum_size=8,
        batch_size=2,
        n_step=1,
        clear_on_stage_change=True,
    )
    config = _config(
        updates_per_collection=0.5,
        replay=replay,
        network=network,
        exploration=exploration,
    )
    state, executables = _create(config, jnp)
    state, action = _first_action(state, executables)
    state, action, _ = _consume(state, executables, _transition(jnp, action), jnp)
    assert int(state.replay.size) == 2

    transition = _transition(jnp, action, step=1)
    expected_action, expected_exploration, expected_key = collection_action(
        Actor(action_dim=2, config=network),
        {"params": state.actor.params, "batch_stats": state.actor.batch_stats},
        transition.post_reset_observation,
        state.exploration,
        state.key,
        truncated_zeta_cdf(exploration.zeta_exponent, exploration.maximum_noise_repeat),
        stochastic=True,
    )
    jax.block_until_ready(expected_action)

    state, next_action, _ = _consume(
        state,
        executables,
        transition,
        jnp,
        target_voc_scale=0.1,
        stage_changed=True,
    )

    assert int(state.replay.size) == 2
    np.testing.assert_array_equal(np.asarray(next_action), np.asarray(expected_action))
    np.testing.assert_array_equal(
        np.asarray(state.exploration.repeat_count),
        np.asarray(expected_exploration.repeat_count),
    )
    np.testing.assert_array_equal(
        jax.random.key_data(state.key),
        jax.random.key_data(expected_key),
    )


def test_nonfinite_critic_skips_only_adam_but_keeps_attempt_side_effects():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import OptimizationConfig

    optimization = OptimizationConfig(
        loss_scale_initial=8.0,
        loss_scale_growth_factor=2.0,
        loss_scale_backoff_factor=0.5,
        loss_scale_growth_interval=2,
    )
    config = _config(
        updates_per_collection=1.0,
        mixed_precision=True,
        reward_normalization=False,
        optimization=optimization,
    )
    state, executables = _create(config, jnp)
    state, action = _first_action(state, executables)
    state = state.replace(
        target_critic=state.target_critic.replace(
            params=jax.tree.map(jnp.zeros_like, state.target_critic.params),
        )
    )
    critic_params = _tree_numpy(jax, state.critic.params)
    critic_optimizer = _tree_numpy(jax, state.critic.optimizer_state)
    actor_batch_stats = _tree_numpy(jax, state.actor.batch_stats)
    target_batch_stats = _tree_numpy(jax, state.target_critic.batch_stats)
    temperature = float(jnp.exp(state.temperature.log_temperature))

    state, _, metrics = _consume(
        state,
        executables,
        _transition(jnp, action, invalid_action=True),
        jnp,
    )

    assert float(metrics.actor_skipped_fraction) == 0.0
    assert float(metrics.critic_skipped_fraction) == 1.0
    _assert_tree_allclose(jax, state.critic.params, critic_params)
    _assert_tree_equal(jax, state.critic.optimizer_state, critic_optimizer)
    assert int(state.actor.schedule_step) == 1
    assert int(state.temperature.schedule_step) == 1
    assert int(state.critic.schedule_step) == 1
    assert float(state.loss_scale.scale) == 4.0
    assert int(state.loss_scale.finite_steps) == 0
    assert float(jnp.exp(state.temperature.log_temperature)) != temperature
    assert any(
        not np.array_equal(np.asarray(actual), before)
        for actual, before in zip(jax.tree.leaves(state.actor.batch_stats), actor_batch_stats, strict=True)
    )
    assert any(
        not np.array_equal(np.asarray(actual), before)
        for actual, before in zip(jax.tree.leaves(state.target_critic.batch_stats), target_batch_stats, strict=True)
    )
    assert any(np.any(np.asarray(value) != 0.0) for value in jax.tree.leaves(state.target_critic.params))


def test_apply_adam_if_finite_preserves_parameters_and_moments_on_failure():
    jax, jnp = _dependencies()
    optax = pytest.importorskip("optax")

    from flash_chord.training.flash_sac.learner import _apply_adam_if_finite

    optimizer = optax.scale_by_adam()
    params = {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}
    optimizer_state = optimizer.init(params)
    gradients = {"weight": jnp.asarray([jnp.inf, 1.0], dtype=jnp.float32)}
    next_params, next_optimizer_state = jax.jit(
        lambda p, state: _apply_adam_if_finite(
            optimizer,
            p,
            state,
            gradients,
            jnp.asarray(1.0e-3),
            jnp.asarray(False),
        )
    )(params, optimizer_state)

    _assert_tree_equal(jax, next_params, _tree_numpy(jax, params))
    _assert_tree_equal(jax, next_optimizer_state, _tree_numpy(jax, optimizer_state))


def test_compiled_transition_validation_rejects_shape():
    _, jnp = _dependencies()

    state, executables = _create(_config(), jnp)
    transition = _transition(jnp, jnp.zeros((2, 2), dtype=jnp.float32))
    transition = transition.replace(action=jnp.zeros((2, 3), dtype=jnp.float32))

    with pytest.raises(ValueError, match="transition action"):
        _consume(state, executables, transition, jnp)
