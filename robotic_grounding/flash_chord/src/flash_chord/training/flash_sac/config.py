# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration for the native FlashSAC learner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from flash_chord.lifecycle.curriculum import FixedCurriculum


@dataclass(frozen=True)
class NetworkConfig:
    """Actor and distributional double-critic architecture."""

    actor_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_head_mode: str = "upstream"
    residual_mean_head: str = "unit_gain"
    initial_normalized_std: float = 0.1
    action_transform: str = "tanh"
    action_scale: float = 1.0
    critic_blocks: int = 2
    critic_hidden_dim: int = 256
    expansion: int = 4
    critic_count: int = 2
    atom_count: int = 101
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    batch_norm_update_rate: float = 0.01
    batch_norm_epsilon: float = 1.0e-5
    rms_norm_epsilon: float = 1.0e-6
    projection_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        positive_counts = {
            "actor_blocks": self.actor_blocks,
            "actor_hidden_dim": self.actor_hidden_dim,
            "critic_blocks": self.critic_blocks,
            "critic_hidden_dim": self.critic_hidden_dim,
            "expansion": self.expansion,
        }
        invalid_counts = {name: value for name, value in positive_counts.items() if value <= 0}
        if invalid_counts:
            raise ValueError(f"network dimensions and block counts must be positive, got {invalid_counts}")
        if self.critic_count != 2:
            raise ValueError(f"FlashSAC requires exactly two critics, got {self.critic_count}")
        if self.atom_count < 2:
            raise ValueError(f"atom_count must be at least two, got {self.atom_count}")
        if self.actor_head_mode not in {"upstream", "residual"}:
            raise ValueError(f"actor_head_mode must be 'upstream' or 'residual', got {self.actor_head_mode!r}")
        if self.residual_mean_head not in {"unit_gain", "zero_dense"}:
            raise ValueError(f"residual_mean_head must be 'unit_gain' or 'zero_dense', got {self.residual_mean_head!r}")
        if self.actor_head_mode != "residual" and self.residual_mean_head != "unit_gain":
            raise ValueError(
                f"residual_mean_head='zero_dense' requires actor_head_mode='residual', got {self.actor_head_mode!r}"
            )
        if not math.isfinite(self.initial_normalized_std) or self.initial_normalized_std <= 0.0:
            raise ValueError(f"initial_normalized_std must be finite and positive, got {self.initial_normalized_std}")
        if self.action_transform not in {"tanh", "identity"}:
            raise ValueError(f"action_transform must be 'tanh' or 'identity', got {self.action_transform!r}")
        if not math.isfinite(self.action_scale) or self.action_scale <= 0.0:
            raise ValueError(f"action_scale must be positive and finite, got {self.action_scale}")
        if not math.isfinite(self.log_std_min) or not math.isfinite(self.log_std_max):
            raise ValueError("log standard-deviation bounds must be finite")
        if self.log_std_min >= self.log_std_max:
            raise ValueError(
                f"log_std_min must be less than log_std_max, got {self.log_std_min} and {self.log_std_max}"
            )
        initial_log_std = math.log(self.initial_normalized_std)
        if self.actor_head_mode == "residual" and not self.log_std_min < initial_log_std < self.log_std_max:
            raise ValueError(
                "residual initial_normalized_std must have a logarithm strictly inside the log standard-deviation "
                f"bounds, got log({self.initial_normalized_std})={initial_log_std} for "
                f"({self.log_std_min}, {self.log_std_max})"
            )
        if not 0.0 < self.batch_norm_update_rate <= 1.0:
            raise ValueError(f"batch_norm_update_rate must be in (0, 1], got {self.batch_norm_update_rate}")
        epsilons = {
            "batch_norm_epsilon": self.batch_norm_epsilon,
            "rms_norm_epsilon": self.rms_norm_epsilon,
            "projection_epsilon": self.projection_epsilon,
        }
        invalid_epsilons = {name: value for name, value in epsilons.items() if not math.isfinite(value) or value <= 0.0}
        if invalid_epsilons:
            raise ValueError(f"network epsilons must be finite and positive, got {invalid_epsilons}")


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer, update cadence, and target-network settings."""

    learning_rate_initial: float = 3.0e-4
    learning_rate_peak: float = 3.0e-4
    learning_rate_end: float = 1.5e-4
    warmup_fraction: float = 1.0e-6
    decay_fraction: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    loss_scale_initial: float = 65_536.0
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5
    loss_scale_growth_interval: int = 2_000
    actor_update_period: int = 2
    actor_learning_rate_multiplier: float = 1.0
    residual_mean_learning_rate_multiplier: float = 1.0
    behavior_cloning_coefficient: float = 0.0
    target_tau: float = 0.01
    learning_rate_schedule_updates: int | None = None

    def __post_init__(self) -> None:
        learning_rates = (
            self.learning_rate_initial,
            self.learning_rate_peak,
            self.learning_rate_end,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in learning_rates):
            raise ValueError(f"learning rates must be finite and positive, got {learning_rates}")
        if self.learning_rate_schedule_updates is not None and (
            isinstance(self.learning_rate_schedule_updates, bool)
            or not isinstance(self.learning_rate_schedule_updates, int)
            or self.learning_rate_schedule_updates <= 0
        ):
            raise ValueError(
                "learning_rate_schedule_updates must be a positive integer or None, got "
                f"{self.learning_rate_schedule_updates!r}"
            )
        if not 0.0 <= self.warmup_fraction < self.decay_fraction <= 1.0:
            raise ValueError(
                "schedule fractions must satisfy 0 <= warmup < decay <= 1, got "
                f"{self.warmup_fraction} and {self.decay_fraction}"
            )
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError(f"Adam betas must be in [0, 1), got {self.adam_beta1} and {self.adam_beta2}")
        if not math.isfinite(self.adam_epsilon) or self.adam_epsilon <= 0.0:
            raise ValueError(f"adam_epsilon must be finite and positive, got {self.adam_epsilon}")
        if not math.isfinite(self.loss_scale_initial) or self.loss_scale_initial <= 0.0:
            raise ValueError(f"loss_scale_initial must be finite and positive, got {self.loss_scale_initial}")
        if not math.isfinite(self.loss_scale_growth_factor) or self.loss_scale_growth_factor <= 1.0:
            raise ValueError(
                f"loss_scale_growth_factor must be finite and greater than one, got {self.loss_scale_growth_factor}"
            )
        if not math.isfinite(self.loss_scale_backoff_factor) or not 0.0 < self.loss_scale_backoff_factor < 1.0:
            raise ValueError(
                f"loss_scale_backoff_factor must be finite and in (0, 1), got {self.loss_scale_backoff_factor}"
            )
        if self.loss_scale_growth_interval <= 0:
            raise ValueError(f"loss_scale_growth_interval must be positive, got {self.loss_scale_growth_interval}")
        if self.actor_update_period <= 0:
            raise ValueError(f"actor_update_period must be positive, got {self.actor_update_period}")
        if not math.isfinite(self.actor_learning_rate_multiplier) or self.actor_learning_rate_multiplier <= 0.0:
            raise ValueError(
                f"actor_learning_rate_multiplier must be finite and positive, got {self.actor_learning_rate_multiplier}"
            )
        if (
            not math.isfinite(self.residual_mean_learning_rate_multiplier)
            or self.residual_mean_learning_rate_multiplier <= 0.0
        ):
            raise ValueError(
                "residual_mean_learning_rate_multiplier must be finite and positive, got "
                f"{self.residual_mean_learning_rate_multiplier}"
            )
        if not math.isfinite(self.behavior_cloning_coefficient) or self.behavior_cloning_coefficient < 0.0:
            raise ValueError(
                f"behavior_cloning_coefficient must be finite and non-negative, got {self.behavior_cloning_coefficient}"
            )
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError(f"target_tau must be in (0, 1], got {self.target_tau}")

    def schedule_steps(self, total_optimizer_updates: int) -> tuple[int, int]:
        """Return upstream-compatible integer warm-up and decay step counts."""
        if total_optimizer_updates <= 0:
            raise ValueError(f"total_optimizer_updates must be positive, got {total_optimizer_updates}")
        warmup_steps = int(self.warmup_fraction * total_optimizer_updates)
        decay_steps = int(self.decay_fraction * total_optimizer_updates)
        if decay_steps <= warmup_steps:
            raise ValueError(
                f"schedule resolves to non-positive decay span: warmup={warmup_steps}, decay={decay_steps}"
            )
        return warmup_steps, decay_steps


@dataclass(frozen=True)
class ReplayConfig:
    """Fixed-capacity device replay settings."""

    capacity: int = 1_000_000
    minimum_size: int = 100_000
    batch_size: int = 2_048
    n_step: int = 3
    sampling: str = "uniform"
    clear_on_stage_change: bool = False
    observation_storage_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if not 0 < self.minimum_size <= self.capacity:
            raise ValueError(
                f"minimum_size must be in [1, capacity], got {self.minimum_size} for capacity {self.capacity}"
            )
        if not 0 < self.batch_size <= self.capacity:
            raise ValueError(f"batch_size must be in [1, capacity], got {self.batch_size} for capacity {self.capacity}")
        if self.n_step <= 0:
            raise ValueError(f"n_step must be positive, got {self.n_step}")
        if self.sampling != "uniform":
            raise ValueError(f"unsupported replay sampling {self.sampling!r}; expected 'uniform'")
        if self.observation_storage_dtype not in {"float32", "float16"}:
            raise ValueError(
                f"observation_storage_dtype must be 'float32' or 'float16', got {self.observation_storage_dtype!r}"
            )


@dataclass(frozen=True)
class ExplorationConfig:
    """Entropy target and temporally repeated action-noise settings."""

    target_sigma: float = 0.15
    initial_temperature: float = 0.01
    zeta_exponent: float = 2.0
    maximum_noise_repeat: int = 16
    warmup_action: str = "uniform"

    def __post_init__(self) -> None:
        positive_values = {
            "target_sigma": self.target_sigma,
            "initial_temperature": self.initial_temperature,
            "zeta_exponent": self.zeta_exponent,
        }
        invalid_values = {
            name: value for name, value in positive_values.items() if not math.isfinite(value) or value <= 0.0
        }
        if invalid_values:
            raise ValueError(f"exploration values must be finite and positive, got {invalid_values}")
        if self.maximum_noise_repeat <= 0:
            raise ValueError(f"maximum_noise_repeat must be positive, got {self.maximum_noise_repeat}")
        if self.warmup_action not in {"uniform", "policy"}:
            raise ValueError(f"warmup_action must be 'uniform' or 'policy', got {self.warmup_action!r}")

    def target_entropy(self, action_dim: int, action_scale: float = 1.0) -> float:
        """Return the Gaussian-entropy target in scaled environment-action coordinates."""
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if not math.isfinite(action_scale) or action_scale <= 0.0:
            raise ValueError(f"action_scale must be positive and finite, got {action_scale}")
        return 0.5 * action_dim * math.log(2.0 * math.pi * math.e * self.target_sigma**2) + (
            action_dim * math.log(action_scale)
        )


@dataclass(frozen=True)
class RewardNormalizationConfig:
    """Adaptive discounted-return scaling used by the categorical critic."""

    enabled: bool = True
    normalized_return_bound: float = 5.0
    epsilon: float = 1.0e-8
    running_moments_epsilon: float = 1.0e-4

    def __post_init__(self) -> None:
        values = {
            "normalized_return_bound": self.normalized_return_bound,
            "epsilon": self.epsilon,
            "running_moments_epsilon": self.running_moments_epsilon,
        }
        invalid_values = {name: value for name, value in values.items() if not math.isfinite(value) or value <= 0.0}
        if invalid_values:
            raise ValueError(f"reward-normalization values must be finite and positive, got {invalid_values}")

    @property
    def value_support(self) -> tuple[float, float]:
        """Symmetric categorical critic support."""
        return -self.normalized_return_bound, self.normalized_return_bound


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint cadence and resume-state policy."""

    save_interval_environment_steps: int = 5_000_000
    save_replay: bool = False
    load_replay_path: str | None = None
    load_optimizer: bool = True
    load_reward_normalizer: bool = True

    def __post_init__(self) -> None:
        if self.save_interval_environment_steps <= 0:
            raise ValueError(
                f"save_interval_environment_steps must be positive, got {self.save_interval_environment_steps}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Complete collection, optimization, curriculum, and logging configuration."""

    seed: int = 42
    world_count: int = 1_024
    total_environment_steps: int = 50_000_896
    updates_per_collection: float = 2.0
    discount: float = 0.99
    mixed_precision: bool = True
    compute_dtype: str = "float16"
    device: str = "cuda:0"
    resume: bool = False
    checkpoint_path: str | None = None
    wandb_entity: str = "nvidia-isaac"
    wandb_project: str = "v2d-flash-chord"
    experiment_name: str = "sharpa"
    run_name: str = "flash_sac"
    curriculum: FixedCurriculum | None = None
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    reward_normalization: RewardNormalizationConfig = field(default_factory=RewardNormalizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        if self.world_count <= 0:
            raise ValueError(f"world_count must be positive, got {self.world_count}")
        if self.total_environment_steps <= 0:
            raise ValueError(f"total_environment_steps must be positive, got {self.total_environment_steps}")
        if self.total_environment_steps % self.world_count != 0:
            raise ValueError(
                "total_environment_steps must be divisible by world_count, got "
                f"{self.total_environment_steps} and {self.world_count}"
            )
        if not math.isfinite(self.updates_per_collection) or self.updates_per_collection <= 0.0:
            raise ValueError(f"updates_per_collection must be finite and positive, got {self.updates_per_collection}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        if self.compute_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"compute_dtype must be 'float16', 'bfloat16', or 'float32', got {self.compute_dtype!r}")
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume requires checkpoint_path")
        if self.replay.capacity < self.world_count:
            raise ValueError(
                f"replay capacity {self.replay.capacity} must hold at least one {self.world_count}-world collection"
            )
        if (
            self.network.actor_head_mode != "residual"
            and self.optimization.residual_mean_learning_rate_multiplier != 1.0
        ):
            raise ValueError(
                "residual_mean_learning_rate_multiplier requires network.actor_head_mode='residual', got "
                f"{self.network.actor_head_mode!r}"
            )
        if self.total_environment_steps < self.replay.minimum_size:
            raise ValueError(
                f"total_environment_steps {self.total_environment_steps} cannot reach replay minimum "
                f"{self.replay.minimum_size}"
            )
        self.optimization.schedule_steps(self.learning_rate_schedule_updates)

    @property
    def collection_count(self) -> int:
        """Number of vectorized environment interactions in the run."""
        return self.total_environment_steps // self.world_count

    @property
    def total_optimizer_updates(self) -> int:
        """Number of complete optimizer updates produced by the update accumulator."""
        return int(self.collection_count * self.updates_per_collection)

    @property
    def learning_rate_schedule_updates(self) -> int:
        """Optimizer-clock horizon used by every learning-rate schedule."""
        configured = self.optimization.learning_rate_schedule_updates
        return self.total_optimizer_updates if configured is None else configured

    @property
    def schedule_steps(self) -> tuple[int, int]:
        """Warm-up and decay lengths for the effective learning-rate horizon."""
        return self.optimization.schedule_steps(self.learning_rate_schedule_updates)
