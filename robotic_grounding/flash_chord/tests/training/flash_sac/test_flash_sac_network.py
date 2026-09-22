# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural and numerical tests for the native FlashSAC networks."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

_ORACLE_PATH = Path(__file__).with_name("fixtures") / "network_oracle.json"


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    return jax, jnp


def _small_config():
    from flash_chord.training.flash_sac.config import NetworkConfig

    return NetworkConfig(
        actor_blocks=1,
        actor_hidden_dim=8,
        critic_blocks=1,
        critic_hidden_dim=8,
        expansion=2,
        atom_count=11,
    )


def _semantic_values(label, shape):
    offset = (sum((index + 1) * ord(character) for index, character in enumerate(label)) % 29 - 14) * 0.01
    value = np.linspace(offset - 0.3, offset + 0.3, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if label.endswith("running_variance"):
        value = np.abs(value) + 0.5
    return value


def _inject_semantic_values(variables, jnp):
    def inject(value, path):
        if isinstance(value, Mapping):
            return {name: inject(child, (*path, name)) for name, child in value.items()}
        return jnp.asarray(_semantic_values("/".join(path), value.shape))

    return inject(variables, ())


def _nested(mapping, path):
    value = mapping
    for name in path.split("/"):
        value = value[name]
    return value


def test_unit_batch_norm_matches_upstream_running_statistics():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import UnitBatchNorm

    module = UnitBatchNorm(update_rate=0.25, epsilon=1.0e-5)
    initial = module.init(jax.random.key(0), jnp.zeros((3, 2)), training=False)
    value = jnp.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    output, state = module.apply(initial, value, training=True, mutable=["batch_stats"])

    mean = np.asarray([3.0, 6.0])
    biased_variance = np.asarray([8.0 / 3.0, 32.0 / 3.0])
    expected = (np.asarray(value) - mean) / np.sqrt(biased_variance + 1.0e-5)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(state["batch_stats"]["running_mean"]), [0.75, 1.5])
    np.testing.assert_allclose(np.asarray(state["batch_stats"]["running_variance"]), [1.75, 4.75])


def test_ensemble_batch_norm_keeps_critic_statistics_independent():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import EnsembleUnitBatchNorm

    module = EnsembleUnitBatchNorm(ensemble_size=2, update_rate=1.0, epsilon=1.0e-5)
    initial = module.init(jax.random.key(0), jnp.zeros((2, 3, 1)), training=False)
    value = jnp.asarray([[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]])
    output, state = module.apply(initial, value, training=True, mutable=["batch_stats"])

    np.testing.assert_allclose(np.asarray(output.mean(axis=1)), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(state["batch_stats"]["running_mean"]), [[2.0], [20.0]])
    np.testing.assert_allclose(np.asarray(state["batch_stats"]["running_variance"]), [[1.0], [100.0]])


def test_training_batch_norm_rejects_single_sample():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import UnitBatchNorm

    module = UnitBatchNorm()
    with pytest.raises(ValueError, match="at least two samples"):
        module.init(jax.random.key(0), jnp.zeros((1, 3)), training=True)


def test_parameter_projection_enforces_every_constraint_without_touching_biases():
    _, jnp = _dependencies()
    from flax.core import FrozenDict, freeze

    from flash_chord.training.flash_sac.network import project_unit_parameters

    parameters = freeze(
        {
            "linear": {"unit_kernel": jnp.asarray([[3.0, 0.0], [4.0, 2.0]])},
            "gained": {
                "unit_kernel": jnp.asarray([[3.0, 0.0], [4.0, 2.0]]),
                "gain": jnp.asarray([0.25, -2.0]),
            },
            "dense": {"kernel": jnp.asarray([[3.0, 0.0], [4.0, 2.0]])},
            "ensemble": {"unit_kernel": jnp.asarray([[[3.0], [4.0]], [[0.0], [2.0]]])},
            "norm": {"unit_scale": jnp.asarray([3.0, 0.0]), "unit_bias": jnp.asarray([4.0, 0.0])},
            "ensemble_norm": {
                "unit_scale": jnp.asarray([[3.0, 0.0], [0.0, 2.0]]),
                "unit_bias": jnp.asarray([[4.0, 0.0], [0.0, 0.0]]),
            },
            "rms": {"rms_scale": jnp.asarray([3.0, 4.0])},
            "explicit_bias": jnp.asarray([7.0]),
        }
    )

    projected = project_unit_parameters(parameters)

    assert isinstance(projected, FrozenDict)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(projected["linear"]["unit_kernel"]), axis=-2), 1.0)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(projected["gained"]["unit_kernel"]), axis=-2), 1.0)
    np.testing.assert_array_equal(np.asarray(projected["gained"]["gain"]), [0.25, -2.0])
    np.testing.assert_array_equal(np.asarray(projected["dense"]["kernel"]), [[3.0, 0.0], [4.0, 2.0]])
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(projected["ensemble"]["unit_kernel"]), axis=-2),
        1.0,
    )
    for name in ("norm", "ensemble_norm"):
        scale = np.asarray(projected[name]["unit_scale"])
        bias = np.asarray(projected[name]["unit_bias"])
        np.testing.assert_allclose(np.sum(scale**2 + bias**2, axis=-1), scale.shape[-1], rtol=1.0e-6)
    rms_scale = np.asarray(projected["rms"]["rms_scale"])
    np.testing.assert_allclose(np.sum(rms_scale**2, axis=-1), rms_scale.shape[-1], rtol=1.0e-6)
    np.testing.assert_array_equal(np.asarray(projected["explicit_bias"]), [7.0])
    np.testing.assert_array_equal(np.asarray(parameters["linear"]["unit_kernel"]), [[3.0, 0.0], [4.0, 2.0]])


def test_residual_actor_head_starts_observation_independent_with_requested_standard_deviation():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import NetworkConfig
    from flash_chord.training.flash_sac.network import Actor, project_unit_parameters

    initial_std = 0.12
    config = NetworkConfig(
        actor_blocks=1,
        actor_hidden_dim=8,
        actor_head_mode="residual",
        initial_normalized_std=initial_std,
        critic_blocks=1,
        critic_hidden_dim=8,
        expansion=2,
        atom_count=11,
    )
    observation = jnp.asarray(
        [
            [-1_000.0, 0.0, 1_000.0, 0.25],
            [0.5, -0.75, 2.0, -4.0],
            [7.0, 11.0, -13.0, 17.0],
        ],
        dtype=jnp.float32,
    )
    for compute_dtype in (jnp.float32, jnp.float16):
        actor = Actor(action_dim=3, config=config, dtype=compute_dtype)
        variables = actor.init(jax.random.key(7), observation, training=False)
        variables = {**variables, "params": project_unit_parameters(variables["params"])}
        distribution = actor.apply(variables, observation, training=False)

        np.testing.assert_array_equal(np.asarray(distribution.mean), np.zeros((3, 3), dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(distribution.std),
            np.full((3, 3), initial_std, dtype=np.float32),
        )
    np.testing.assert_array_equal(np.asarray(variables["params"]["mean"]["gain"]), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(variables["params"]["log_std"]["gain"]), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(variables["params"]["mean_bias"]), np.zeros(3))
    for head in ("mean", "log_std"):
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(variables["params"][head]["unit_kernel"]), axis=0),
            1.0,
            rtol=1.0e-6,
        )


def test_zero_dense_residual_head_starts_at_zero_with_immediate_kernel_gradients():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import NetworkConfig
    from flash_chord.training.flash_sac.network import Actor, project_unit_parameters

    config = NetworkConfig(
        actor_blocks=1,
        actor_hidden_dim=8,
        actor_head_mode="residual",
        residual_mean_head="zero_dense",
        initial_normalized_std=0.12,
        critic_blocks=1,
        critic_hidden_dim=8,
        expansion=2,
        atom_count=11,
    )
    observation = jnp.asarray(
        [
            [-1.0, 0.25, 0.5, 2.0],
            [0.5, -0.75, 2.0, -4.0],
            [7.0, 11.0, -13.0, 17.0],
        ],
        dtype=jnp.float32,
    )
    action_cotangent = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

    for compute_dtype in (jnp.float32, jnp.float16):
        actor = Actor(action_dim=3, config=config, dtype=compute_dtype)
        variables = actor.init(jax.random.key(11), observation, training=False)
        parameters = project_unit_parameters(variables["params"])
        variables = {**variables, "params": parameters}
        distribution = actor.apply(variables, observation, training=False)

        np.testing.assert_array_equal(np.asarray(distribution.mean), np.zeros((3, 3), dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(parameters["mean"]["kernel"]), np.zeros((8, 3), dtype=np.float32))
        assert "gain" not in parameters["mean"]
        assert distribution.mean.dtype == jnp.dtype(jnp.float32)

        def mean_objective(candidate):
            candidate_variables = {**variables, "params": candidate}
            mean = actor.apply(candidate_variables, observation, training=False).mean
            return jnp.sum(mean * action_cotangent)

        gradient = jax.grad(mean_objective)(parameters)["mean"]["kernel"]
        assert np.all(np.any(np.asarray(gradient) != 0.0, axis=0))


def test_actor_shapes_log_std_bounds_sampling_and_jit():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import (
        Actor,
        deterministic_action,
        project_unit_parameters,
        sample_squashed_gaussian,
    )

    config = _small_config()
    actor = Actor(action_dim=2, config=config)
    observation = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.0
    variables = actor.init(jax.random.key(0), observation, training=False)
    variables = {**variables, "params": project_unit_parameters(variables["params"])}
    distribution = jax.jit(lambda values, obs: actor.apply(values, obs, training=False))(variables, observation)
    sample = sample_squashed_gaussian(distribution, jax.random.key(1))

    assert distribution.mean.shape == (4, 2)
    assert distribution.std.shape == (4, 2)
    assert np.all(np.asarray(distribution.log_std) >= config.log_std_min)
    assert np.all(np.asarray(distribution.log_std) <= config.log_std_max)
    assert sample.action.shape == (4, 2)
    assert sample.log_probability.shape == (4,)
    assert np.all(np.abs(np.asarray(sample.action)) <= 1.0)
    np.testing.assert_array_equal(
        np.asarray(deterministic_action(distribution)),
        np.asarray(jnp.tanh(distribution.mean)),
    )
    assert np.isfinite(np.asarray(sample.log_probability)).all()


def test_mixed_precision_limits_float16_to_matrix_multiplication():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import (
        Actor,
        Critic,
        UnitLinear,
        sample_squashed_gaussian,
    )

    linear = UnitLinear(features=4, dtype=jnp.float16)
    linear_input = jnp.ones((2, 3), dtype=jnp.float32)
    linear_variables = linear.init(jax.random.key(0), linear_input)
    linear_output = jax.jit(linear.apply)(linear_variables, linear_input)
    linear_jaxpr = str(jax.make_jaxpr(linear.apply)(linear_variables, linear_input))
    assert linear_output.dtype == jnp.float32
    assert "new_dtype=float16" in linear_jaxpr
    assert "dot_general" in linear_jaxpr

    config = _small_config()
    actor = Actor(action_dim=2, config=config, dtype=jnp.float16)
    observation = jnp.ones((4, 3), dtype=jnp.float32)
    actor_variables = actor.init(jax.random.key(1), observation, training=False)
    distribution = actor.apply(actor_variables, observation, training=False)
    sample = sample_squashed_gaussian(distribution, jax.random.key(2))
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(distribution))
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(sample))
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(actor_variables))

    critic = Critic(value_min=-5.0, value_max=5.0, config=config, dtype=jnp.float16)
    action = jnp.zeros((4, 2), dtype=jnp.float32)
    critic_variables = critic.init(jax.random.key(3), observation, action, training=False)
    critic_output = critic.apply(critic_variables, observation, action, training=False)
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(critic_output))
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(critic_variables))


def test_mixed_precision_critic_adds_categorical_bias_in_float32():
    jax, jnp = _dependencies()
    from flax.core import freeze, unfreeze

    from flash_chord.training.flash_sac.network import Critic

    config = _small_config()
    critic = Critic(value_min=-5.0, value_max=5.0, config=config, dtype=jnp.float16)
    observation = jnp.zeros((4, 3), dtype=jnp.float32)
    action = jnp.zeros((4, 2), dtype=jnp.float32)
    variables = critic.init(jax.random.key(3), observation, action, training=False)
    params = unfreeze(variables["params"])
    params["categorical"]["unit_kernel"] = jnp.zeros_like(params["categorical"]["unit_kernel"])
    bias = jnp.asarray(
        [
            np.linspace(-1.0003, 0.9997, config.atom_count, dtype=np.float32),
            np.linspace(0.9997, -1.0003, config.atom_count, dtype=np.float32),
        ]
    )
    params["categorical_bias"] = bias
    output = critic.apply(
        {**variables, "params": freeze(params)},
        observation,
        action,
        training=False,
    )

    assert output.logits.dtype == jnp.dtype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(output.logits), np.broadcast_to(np.asarray(bias)[:, None, :], (2, 4, 11)))


def test_explicit_noise_sample_has_exact_tanh_corrected_log_probability():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import ActorDistribution, squashed_gaussian_from_noise

    distribution = ActorDistribution(
        mean=jnp.asarray([[0.2, -0.4]]),
        log_std=jnp.log(jnp.asarray([[0.5, 1.5]])),
        std=jnp.asarray([[0.5, 1.5]]),
    )
    noise = jnp.asarray([[0.3, -0.2]])
    sample = squashed_gaussian_from_noise(distribution, noise)

    raw_action = np.asarray(distribution.mean + distribution.std * noise)
    normal_log_probability = -0.5 * (
        np.square(np.asarray(noise)) + 2.0 * np.asarray(distribution.log_std) + math.log(2.0 * math.pi)
    )
    jacobian = np.log(1.0 - np.square(np.tanh(raw_action)))
    expected = np.sum(normal_log_probability - jacobian, axis=-1)
    np.testing.assert_allclose(np.asarray(sample.raw_action), raw_action)
    np.testing.assert_allclose(np.asarray(sample.action), np.tanh(raw_action))
    np.testing.assert_allclose(np.asarray(sample.log_probability), expected, rtol=1.0e-6)


def test_identity_gaussian_preserves_unbounded_raw_actions_and_exact_density():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import ActorDistribution, policy_sample_from_noise

    distribution = ActorDistribution(
        mean=jnp.asarray([[2.0, -3.0]]),
        log_std=jnp.log(jnp.asarray([[0.5, 1.5]])),
        std=jnp.asarray([[0.5, 1.5]]),
    )
    noise = jnp.asarray([[0.3, -0.2]])
    sample = policy_sample_from_noise(
        distribution,
        noise,
        action_transform="identity",
        action_scale=2.0,
    )

    raw_action = np.asarray(distribution.mean + distribution.std * noise)
    normal_log_probability = -0.5 * (
        np.square(np.asarray(noise)) + 2.0 * np.asarray(distribution.log_std) + math.log(2.0 * math.pi)
    )
    expected = np.sum(normal_log_probability - math.log(2.0), axis=-1)
    np.testing.assert_allclose(np.asarray(sample.raw_action), raw_action)
    np.testing.assert_allclose(np.asarray(sample.action), raw_action * 2.0)
    assert np.max(np.abs(np.asarray(sample.action))) > 1.0
    np.testing.assert_allclose(np.asarray(sample.log_probability), expected, rtol=1.0e-6)


def test_stable_tanh_jacobian_is_finite_for_extreme_inputs():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import safe_tanh_log_det_jacobian

    result = safe_tanh_log_det_jacobian(jnp.asarray([-1_000.0, 0.0, 1_000.0]))

    assert np.isfinite(np.asarray(result)).all()
    assert float(result[1]) == pytest.approx(0.0)


def test_critic_shapes_categorical_probabilities_and_support_expectation():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import Critic, project_unit_parameters

    config = _small_config()
    critic = Critic(value_min=-5.0, value_max=5.0, config=config)
    observation = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 10.0
    action = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 10.0
    variables = critic.init(jax.random.key(0), observation, action, training=False)
    variables = {**variables, "params": project_unit_parameters(variables["params"])}
    output = jax.jit(lambda values, obs, act: critic.apply(values, obs, act, training=False))(
        variables,
        observation,
        action,
    )

    assert output.q.shape == (2, 4)
    assert output.logits.shape == (2, 4, 11)
    assert output.log_probability.shape == (2, 4, 11)
    probabilities = np.exp(np.asarray(output.log_probability))
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, rtol=1.0e-6)
    support = np.linspace(-5.0, 5.0, 11)
    np.testing.assert_allclose(np.asarray(output.q), np.sum(probabilities * support, axis=-1), rtol=1.0e-6)
    assert np.all(np.asarray(output.q) >= -5.0)
    assert np.all(np.asarray(output.q) <= 5.0)


def test_ensemble_initialization_uses_independent_members():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.network import EnsembleUnitLinear

    module = EnsembleUnitLinear(ensemble_size=2, features=3)
    variables = module.init(jax.random.key(0), jnp.zeros((2, 4, 5)))
    kernel = np.asarray(variables["params"]["unit_kernel"])

    assert kernel.shape == (2, 5, 3)
    assert not np.array_equal(kernel[0], kernel[1])


def test_mixer_network_parameter_counts_and_critic_context_shape():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import NetworkConfig
    from flash_chord.training.flash_sac.network import Actor, Critic

    config = NetworkConfig()
    actor_variables = Actor(action_dim=56, config=config).init(
        jax.random.key(0),
        jnp.zeros((2, 722)),
        training=False,
    )
    critic_variables = Critic(value_min=-5.0, value_max=5.0, config=config).init(
        jax.random.key(1),
        jnp.zeros((2, 725)),
        jnp.zeros((2, 56)),
        training=False,
    )

    actor_parameter_count = sum(value.size for value in jax.tree.leaves(actor_variables["params"]))
    critic_parameter_count = sum(value.size for value in jax.tree.leaves(critic_variables["params"]))
    assert actor_parameter_count == 373_140
    assert critic_parameter_count == 2_562_814


def test_actor_matches_frozen_upstream_torch_oracle_in_eval_and_training():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import NetworkConfig
    from flash_chord.training.flash_sac.network import Actor

    oracle = json.loads(_ORACLE_PATH.read_text())
    assert oracle["upstream_commit"] == "87edc9061150ae9e962dd84e6544e27a1554b3ab"
    config = NetworkConfig(
        actor_blocks=1,
        actor_hidden_dim=4,
        actor_head_mode="upstream",
        critic_blocks=1,
        critic_hidden_dim=4,
        atom_count=5,
    )
    actor = Actor(action_dim=2, config=config)
    observation = jnp.asarray(oracle["actor_observation"])
    with jax.default_matmul_precision("high"):
        variables = actor.init(jax.random.key(0), observation, training=False)
        variables = _inject_semantic_values(variables, jnp)
        evaluation = actor.apply(variables, observation, training=False)
        training, state = actor.apply(variables, observation, training=True, mutable=["batch_stats"])

    np.testing.assert_allclose(evaluation.mean, oracle["actor_eval_mean"], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(evaluation.std, oracle["actor_eval_std"], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(training.mean, oracle["actor_train_mean"], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(training.std, oracle["actor_train_std"], rtol=2.0e-5, atol=2.0e-6)
    for path, expected in oracle["actor_train_batch_stats"].items():
        np.testing.assert_allclose(_nested(state, path), expected, rtol=2.0e-5, atol=2.0e-6)


def test_critic_matches_frozen_upstream_torch_oracle_in_eval_and_training():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import NetworkConfig
    from flash_chord.training.flash_sac.network import Critic

    oracle = json.loads(_ORACLE_PATH.read_text())
    config = NetworkConfig(actor_blocks=1, actor_hidden_dim=4, critic_blocks=1, critic_hidden_dim=4, atom_count=5)
    critic = Critic(value_min=-5.0, value_max=5.0, config=config)
    observation = jnp.asarray(oracle["critic_observation"])
    action = jnp.asarray(oracle["action"])
    with jax.default_matmul_precision("high"):
        variables = critic.init(jax.random.key(0), observation, action, training=False)
        variables = _inject_semantic_values(variables, jnp)
        evaluation = critic.apply(variables, observation, action, training=False)
        training, state = critic.apply(
            variables,
            observation,
            action,
            training=True,
            mutable=["batch_stats"],
        )

    np.testing.assert_allclose(evaluation.q, oracle["critic_eval_q"], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(
        evaluation.log_probability,
        oracle["critic_eval_log_probability"],
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(training.q, oracle["critic_train_q"], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(
        training.log_probability,
        oracle["critic_train_log_probability"],
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    for path, expected in oracle["critic_train_batch_stats"].items():
        np.testing.assert_allclose(_nested(state, path), expected, rtol=2.0e-5, atol=2.0e-6)
