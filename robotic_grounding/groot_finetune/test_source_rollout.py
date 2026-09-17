# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dynamics-source rollout labels."""

from __future__ import annotations

import numpy as np
import pytest

from groot_finetune.source_rollout import (
    configure_timeout_only_terminations,
    snapshot_applied_action_target,
    source_success_from_terminations,
    summarize_trajectory_diversity,
    termination_reasons_for_env,
    validate_terminal_action_target,
)


class _TerminationCfg:
    def __init__(self, *, time_out: bool = False) -> None:
        self.time_out = time_out


def test_source_collection_config_is_timeout_only() -> None:
    class Terminations:
        timeout = _TerminationCfg(time_out=True)
        tracking_error = _TerminationCfg()
        safety = _TerminationCfg()

        def __init__(self) -> None:
            self.timeout = self.__class__.timeout
            self.tracking_error = self.__class__.tracking_error
            self.safety = self.__class__.safety

    cfg = Terminations()
    assert configure_timeout_only_terminations(cfg) == "timeout"
    assert cfg.timeout is not None
    assert cfg.tracking_error is None
    assert cfg.safety is None


def test_source_collection_requires_one_timeout() -> None:
    class Terminations:
        def __init__(self) -> None:
            self.failure = _TerminationCfg()

    with pytest.raises(ValueError, match="exactly one"):
        configure_timeout_only_terminations(Terminations())


def test_source_success_is_timeout_only() -> None:
    terms = ["timeout", "robot_state_diverged"]
    assert source_success_from_terminations(terms, np.asarray([True, False]), "timeout")
    assert not source_success_from_terminations(
        terms, np.asarray([False, True]), "timeout"
    )
    assert source_success_from_terminations(terms, np.asarray([True, True]), "timeout")


def test_termination_reasons_use_active_iterable_terms() -> None:
    class Manager:
        active_terms = ["timeout", "safety"]

        @staticmethod
        def get_active_iterable_terms(env_index: int) -> list[tuple[str, list[float]]]:
            assert env_index == 3
            return [("timeout", [1.0]), ("safety", [0.0])]

    assert termination_reasons_for_env(Manager(), 3) == ("timeout",)


class _AutoResetActionTerm:
    """Minimal action term reproducing IsaacLab's post-terminal reset ordering."""

    def __init__(self) -> None:
        self.processed_actions = _FakeTensor(np.zeros((1, 3), dtype=np.float32))
        self.last_applied_actions = _FakeTensor(np.zeros((1, 3), dtype=np.float32))

    def apply(self, target: np.ndarray) -> None:
        self.processed_actions.value[...] = target
        self.last_applied_actions.value[...] = self.processed_actions.value

    def reset_before_step_returns(self) -> None:
        self.processed_actions.value.fill(0.0)


class _FakeTensor:
    """Small tensor-shaped object implementing the export helper's read chain."""

    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __getitem__(self, key: object) -> "_FakeTensor":
        return _FakeTensor(self.value[key])

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


def test_terminal_auto_reset_cannot_replace_last_applied_target() -> None:
    term = _AutoResetActionTerm()
    previous = np.asarray([0.40, -0.20, 0.70], dtype=np.float32)
    terminal = np.asarray([[0.41, -0.19, 0.69]], dtype=np.float32)
    term.apply(terminal)
    term.reset_before_step_returns()

    assert np.count_nonzero(term.processed_actions.value) == 0
    captured = snapshot_applied_action_target(term, np.asarray([0, 1, 2]))[0]
    np.testing.assert_allclose(captured, terminal[0])
    action_trace = np.stack([previous, captured])
    assert validate_terminal_action_target(action_trace) < 0.02


def test_refuses_processed_or_raw_action_fallback() -> None:
    class UnsafeTerm:
        processed_actions = np.ones((1, 3))
        raw_actions = np.ones((1, 3))

    with pytest.raises(TypeError, match="last_applied_actions"):
        snapshot_applied_action_target(UnsafeTerm(), slice(None))


def test_rejects_reset_zero_terminal_discontinuity() -> None:
    contaminated = np.asarray([[0.40, -0.20, 0.70], [0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="auto-reset contamination"):
        validate_terminal_action_target(contaminated)


def test_summarizes_phase_aligned_trajectory_diversity() -> None:
    states = np.stack(
        [np.zeros((4, 3), dtype=np.float32), np.full((4, 3), 0.02, dtype=np.float32)]
    )
    actions = np.stack(
        [np.zeros((4, 3), dtype=np.float32), np.full((4, 3), 0.04, dtype=np.float32)]
    )
    poses = np.zeros((2, 4, 1, 7), dtype=np.float32)
    poses[..., 3] = 1.0
    poses[1, ..., 0] = 0.02

    summary = summarize_trajectory_diversity(states, actions, poses)

    assert summary["episode_count"] == 2
    assert summary["horizon"] == 4
    assert summary["state_cross_episode_std_mean_rad"] == pytest.approx(0.01)
    assert summary["action_cross_episode_std_mean_rad"] == pytest.approx(0.02)
    assert summary["initial_object_position_cross_episode_std_mean_m"] == pytest.approx(
        0.01 / 3.0
    )


def test_diversity_summary_rejects_misaligned_episodes() -> None:
    with pytest.raises(ValueError, match="episode/time shapes differ"):
        summarize_trajectory_diversity(
            np.zeros((2, 4, 3), dtype=np.float32),
            np.zeros((2, 3, 3), dtype=np.float32),
        )
