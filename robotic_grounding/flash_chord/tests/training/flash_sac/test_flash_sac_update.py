# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch-oracle and JIT tests for native FlashSAC update math."""

import json
from pathlib import Path

import numpy as np
import pytest

_ORACLE_PATH = Path(__file__).with_name("fixtures") / "update_oracle.json"


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    return jax, jnp


def _oracle():
    oracle = json.loads(_ORACLE_PATH.read_text())
    assert oracle["upstream_commit"] == "87edc9061150ae9e962dd84e6544e27a1554b3ab"
    return oracle


@pytest.mark.parametrize("compiled", [False, True])
def test_minimum_q_selects_complete_distribution_and_ties_choose_first(compiled):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import select_min_q_log_probability

    oracle = _oracle()["minimum_q"]
    select = jax.jit(select_min_q_log_probability) if compiled else select_min_q_log_probability
    selected = select(
        jnp.asarray(oracle["q"], dtype=jnp.float32),
        jnp.asarray(oracle["log_probability"], dtype=jnp.float32),
    )

    np.testing.assert_allclose(np.asarray(selected), oracle["selected"], rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_array_equal(np.asarray(selected[2]), np.asarray(oracle["log_probability"])[0, 2])


@pytest.mark.parametrize("compiled", [False, True])
def test_categorical_target_matches_torch_and_is_gradient_stopped(compiled):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import categorical_td_target

    oracle = _oracle()["categorical_target"]

    def project(log_probability):
        return categorical_td_target(
            log_probability,
            jnp.asarray(oracle["reward"], dtype=jnp.float32),
            jnp.asarray(oracle["terminated"], dtype=jnp.bool_),
            jnp.asarray(oracle["temperature_log_probability"], dtype=jnp.float32),
            bootstrap_discount=oracle["bootstrap_discount"],
            value_min=oracle["value_min"],
            value_max=oracle["value_max"],
        )

    log_probability = jnp.asarray(oracle["target_log_probability"], dtype=jnp.float32)
    project = jax.jit(project) if compiled else project
    probability = project(log_probability)

    np.testing.assert_allclose(np.asarray(probability), oracle["probability"], rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(probability.sum(axis=-1)), 1.0, atol=1.0e-7)
    gradient = jax.grad(lambda value: project(value).sum())(log_probability)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros_like(np.asarray(log_probability)))


def test_categorical_cross_entropy_averages_critics_and_batch():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import categorical_cross_entropy

    target = jnp.asarray([[0.25, 0.75], [1.0, 0.0]], dtype=jnp.float32)
    predicted = jnp.log(
        jnp.asarray(
            [
                [[0.4, 0.6], [0.8, 0.2]],
                [[0.3, 0.7], [0.5, 0.5]],
            ],
            dtype=jnp.float32,
        )
    )
    expected = -np.mean(
        [
            0.25 * np.log(0.4) + 0.75 * np.log(0.6),
            np.log(0.8),
            0.25 * np.log(0.3) + 0.75 * np.log(0.7),
            np.log(0.5),
        ]
    )
    assert float(categorical_cross_entropy(target, predicted)) == pytest.approx(expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_actor_objective_matches_torch_oracle_and_gradient_boundaries(compiled):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import actor_loss

    oracle = _oracle()["actor"]

    def objective(log_probability, minimum_q, action, temperature):
        return actor_loss(
            log_probability,
            minimum_q,
            action,
            jnp.asarray(oracle["replay_action"], dtype=jnp.float32),
            temperature,
            behavior_cloning_coefficient=oracle["behavior_cloning_coefficient"],
        )

    objective = jax.jit(objective) if compiled else objective
    inputs = (
        jnp.asarray(oracle["log_probability"], dtype=jnp.float32),
        jnp.asarray(oracle["minimum_q"], dtype=jnp.float32),
        jnp.asarray(oracle["action"], dtype=jnp.float32),
        jnp.asarray(oracle["temperature"], dtype=jnp.float32),
    )
    result = objective(*inputs)

    assert float(result.loss) == pytest.approx(oracle["loss"], rel=1.0e-6)
    assert float(result.entropy) == pytest.approx(oracle["entropy"])
    assert float(result.mean_action) == pytest.approx(oracle["mean_action"])
    assert float(result.behavior_cloning_loss) == pytest.approx(oracle["behavior_cloning_loss"])

    q_gradient, temperature_gradient = jax.grad(
        lambda minimum_q, temperature: objective(inputs[0], minimum_q, inputs[2], temperature).loss,
        argnums=(0, 1),
    )(inputs[1], inputs[3])
    np.testing.assert_allclose(np.asarray(q_gradient), -1.0 / 3.0, atol=1.0e-7)
    assert float(temperature_gradient) == 0.0


@pytest.mark.parametrize("compiled", [False, True])
def test_temperature_objective_matches_torch_value_loss_and_gradient(compiled):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import temperature_loss

    oracle = _oracle()["temperature"]

    def objective(log_temperature):
        return temperature_loss(
            log_temperature,
            jnp.asarray(oracle["entropy"], dtype=jnp.float32),
            oracle["target_entropy"],
        )

    objective = jax.jit(objective) if compiled else objective
    log_temperature = jnp.asarray([oracle["log_temperature"]], dtype=jnp.float32)
    result = objective(log_temperature)
    gradient = jax.grad(lambda value: objective(value).loss)(log_temperature)

    assert result.loss.shape == ()
    assert result.value.shape == ()
    assert float(result.value) == pytest.approx(oracle["value"], rel=1.0e-6)
    assert float(result.loss) == pytest.approx(oracle["loss"], rel=1.0e-6)
    np.testing.assert_allclose(np.asarray(gradient), [oracle["log_temperature_gradient"]], rtol=1.0e-6)


def test_learning_rate_matches_upstream_schedule_at_boundaries():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import OptimizationConfig
    from flash_chord.training.flash_sac.update import learning_rate

    oracle = _oracle()["schedule"]
    config = OptimizationConfig(
        learning_rate_initial=oracle["initial"],
        learning_rate_peak=oracle["peak"],
        learning_rate_end=oracle["end"],
        warmup_fraction=oracle["warmup_steps"] / oracle["decay_steps"],
        decay_fraction=1.0,
    )
    schedule = jax.jit(lambda step: learning_rate(step, config, oracle["decay_steps"]))
    values = schedule(jnp.asarray(oracle["steps"], dtype=jnp.int32))

    np.testing.assert_allclose(np.asarray(values), oracle["values"], rtol=1.0e-6, atol=1.0e-10)


def test_actor_and_temperature_schedule_clocks_advance_only_when_updated():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import OptimizationConfig
    from flash_chord.training.flash_sac.update import learning_rate

    config = OptimizationConfig(
        learning_rate_initial=1.0e-4,
        learning_rate_peak=4.0e-4,
        learning_rate_end=5.0e-5,
        warmup_fraction=0.1,
        decay_fraction=1.0,
        actor_update_period=2,
    )
    global_updates = jnp.arange(8, dtype=jnp.int32)
    critic_clock = global_updates
    delayed_clock = global_updates // config.actor_update_period
    critic_rate = learning_rate(critic_clock, config, 100)
    delayed_rate = learning_rate(delayed_clock, config, 100)

    assert delayed_clock.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert float(delayed_rate[-1]) < float(critic_rate[-1])


def test_ema_matches_oracle_for_parameter_pytree():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import ema_parameters

    oracle = _oracle()["ema"]
    target = {"layer": jnp.asarray(oracle["target"], dtype=jnp.float32)}
    source = {"layer": jnp.asarray(oracle["source"], dtype=jnp.float32)}
    result = jax.jit(lambda old, new: ema_parameters(old, new, oracle["tau"]))(target, source)

    np.testing.assert_allclose(np.asarray(result["layer"]), oracle["result"], atol=1.0e-7)


def test_shared_loss_scale_matches_pytorch_growth_and_backoff_timing():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import OptimizationConfig
    from flash_chord.training.flash_sac.update import create_loss_scale, update_loss_scale

    config = OptimizationConfig(
        loss_scale_initial=8.0,
        loss_scale_growth_factor=2.0,
        loss_scale_backoff_factor=0.5,
        loss_scale_growth_interval=3,
    )

    @jax.jit
    def advance(state, finite_sequence):
        return jax.lax.scan(
            lambda current, finite: (update_loss_scale(current, finite, config, enabled=True), None),
            state,
            finite_sequence,
        )[0]

    state = create_loss_scale(config, enabled=True)
    state = advance(state, jnp.asarray([True, True]))
    assert float(state.scale) == 8.0
    assert int(state.finite_steps) == 2
    state = advance(state, jnp.asarray([True]))
    assert float(state.scale) == 16.0
    assert int(state.finite_steps) == 0
    state = advance(state, jnp.asarray([False]))
    assert float(state.scale) == 8.0
    assert int(state.finite_steps) == 0

    inert = create_loss_scale(config, enabled=False)
    inert = update_loss_scale(inert, jnp.asarray(False), config, enabled=False)
    assert float(inert.scale) == 1.0
    assert int(inert.finite_steps) == 0


def test_gradient_unscale_promotes_float32_and_finite_check_covers_tree():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.config import OptimizationConfig
    from flash_chord.training.flash_sac.update import (
        create_loss_scale,
        gradients_are_finite,
        unscale_gradients,
    )

    state = create_loss_scale(OptimizationConfig(loss_scale_initial=8.0), enabled=True)

    @jax.jit
    def process(gradients):
        gradients = unscale_gradients(state, gradients)
        return gradients, gradients_are_finite(gradients)

    gradients, finite = process(
        {
            "first": jnp.asarray([8.0, -16.0], dtype=jnp.float16),
            "second": (jnp.asarray([24.0], dtype=jnp.float32),),
        }
    )
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(gradients))
    np.testing.assert_allclose(np.asarray(gradients["first"]), [1.0, -2.0])
    np.testing.assert_allclose(np.asarray(gradients["second"][0]), [3.0])
    assert bool(finite)

    _, finite = process({"first": jnp.asarray([jnp.inf]), "second": (jnp.asarray([1.0]),)})
    assert not bool(finite)


def test_selection_projection_and_cross_entropy_compose_in_one_jit():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import (
        categorical_cross_entropy,
        categorical_td_target,
        select_min_q_log_probability,
    )

    oracle = _oracle()["minimum_q"]
    q = jnp.asarray(oracle["q"], dtype=jnp.float32)
    target_log_probability = jnp.asarray(oracle["log_probability"], dtype=jnp.float32)
    predicted_log_probability = target_log_probability

    @jax.jit
    def loss(q_value, target_log, predicted_log):
        selected = select_min_q_log_probability(q_value, target_log)
        target = categorical_td_target(
            selected,
            jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32),
            jnp.asarray([False, False, True, False], dtype=jnp.bool_),
            jnp.zeros((4,), dtype=jnp.float32),
            bootstrap_discount=0.9**3,
            value_min=-5.0,
            value_max=5.0,
        )
        return categorical_cross_entropy(target, predicted_log)

    value = loss(q, target_log_probability, predicted_log_probability)
    assert value.shape == ()
    assert bool(jnp.isfinite(value))


def test_update_helpers_reject_invalid_shapes_and_values():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.update import (
        actor_loss,
        categorical_cross_entropy,
        categorical_td_target,
        ema_parameters,
        select_min_q_log_probability,
        temperature_loss,
    )

    with pytest.raises(ValueError, match="q must have shape"):
        select_min_q_log_probability(jnp.zeros((3, 2)), jnp.zeros((3, 2, 5)))
    with pytest.raises(ValueError, match="reward has shape"):
        categorical_td_target(
            jnp.zeros((2, 3)),
            jnp.zeros((3,)),
            jnp.zeros((2,), dtype=jnp.bool_),
            jnp.zeros((2,)),
            bootstrap_discount=0.9,
            value_min=-1.0,
            value_max=1.0,
        )
    with pytest.raises(ValueError, match="target_probability"):
        categorical_cross_entropy(jnp.zeros((3, 2)), jnp.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="action/replay shapes"):
        actor_loss(
            jnp.zeros((2,)),
            jnp.zeros((2,)),
            jnp.zeros((2, 1)),
            jnp.zeros((2, 2)),
            jnp.asarray(0.1),
            behavior_cloning_coefficient=0.0,
        )
    with pytest.raises(ValueError, match="one scalar"):
        temperature_loss(jnp.zeros((2,)), jnp.zeros((2,)), 0.0)
    with pytest.raises(ValueError, match="tau"):
        ema_parameters({"x": jnp.zeros(1)}, {"x": jnp.ones(1)}, 0.0)
