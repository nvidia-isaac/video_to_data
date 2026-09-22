# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate deployable-policy, learner-state, and optional replay checkpoints."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import flax
import jax
import jax.numpy as jnp

from flash_chord.training.checkpoint import (
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)
from flash_chord.training.flash_sac.exploration import ExplorationState
from flash_chord.training.flash_sac.learner import (
    LearnerState,
    OptimizedNetworkState,
    TargetCriticState,
    TemperatureState,
)
from flash_chord.training.flash_sac.replay import _flush_pending
from flash_chord.training.flash_sac.reward_normalization import RewardNormalizerState
from flash_chord.training.flash_sac.update import LossScaleState

ALGORITHM_METADATA_KEY = "algorithm"
CHECKPOINT_KIND_METADATA_KEY = "checkpoint_kind"
ALGORITHM_NAME = "flash_sac"
CHECKPOINT_SET_METADATA_KEY = "checkpoint_set_id"
ENVIRONMENT_STEPS_METADATA_KEY = "environment_steps"


@flax.struct.dataclass
class ActorInferenceState:
    """World-count-independent actor state required by deterministic or stochastic eval."""

    params: Any
    batch_stats: Any


@flax.struct.dataclass
class TrainingCheckpoint:
    """Compact learner state; replay, transient logging, and simulator-owned state are separate."""

    actor: OptimizedNetworkState
    critic: OptimizedNetworkState
    target_critic: TargetCriticState
    temperature: TemperatureState
    reward_normalizer: RewardNormalizerState
    exploration: ExplorationState
    loss_scale: LossScaleState
    key: jax.Array
    update_credit: jax.Array
    environment_steps: jax.Array
    global_update_step: jax.Array


@dataclass(frozen=True)
class CheckpointPaths:
    """Files emitted at one environment-step checkpoint boundary."""

    policy: Path
    training_state: Path
    replay: Path | None


def actor_inference_state(learner: LearnerState) -> ActorInferenceState:
    """Extract the deployable actor without optimizer or environment-shaped state."""
    return ActorInferenceState(params=learner.actor.params, batch_stats=learner.actor.batch_stats)


def training_checkpoint(learner: LearnerState) -> TrainingCheckpoint:
    """Extract resumable state while excluding replay, transient logging, and simulator observations."""
    return TrainingCheckpoint(
        actor=learner.actor,
        critic=learner.critic,
        target_critic=learner.target_critic,
        temperature=learner.temperature,
        reward_normalizer=learner.reward_normalizer,
        exploration=learner.exploration,
        loss_scale=learner.loss_scale,
        key=learner.key,
        update_credit=learner.update_credit,
        environment_steps=learner.environment_steps,
        global_update_step=learner.global_update_step,
    )


def _metadata(metadata: Mapping[str, object] | None, kind: str) -> dict[str, object]:
    result = dict(metadata or {})
    result[ALGORITHM_METADATA_KEY] = ALGORITHM_NAME
    result[CHECKPOINT_KIND_METADATA_KEY] = kind
    return result


def save_actor_checkpoint(
    path: str | Path,
    actor: ActorInferenceState,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Save one deployable actor-only checkpoint with standard FlashSAC metadata."""
    path = Path(path)
    save_checkpoint(path, actor, metadata=_metadata(metadata, "policy"))
    return path


def save_learner_checkpoint(
    directory: str | Path,
    learner: LearnerState,
    *,
    metadata: Mapping[str, object] | None = None,
    save_replay: bool = False,
) -> CheckpointPaths:
    """Save policy and compact state, with replay in a separate explicitly requested file."""
    directory = Path(directory)
    step = int(jax.device_get(learner.environment_steps))
    policy_path = directory / f"policy_{step}.safetensors"
    state_path = directory / f"state_{step}.safetensors"
    replay_path = directory / f"replay_{step}.safetensors" if save_replay else None
    checkpoint_metadata = dict(metadata or {})
    checkpoint_metadata[CHECKPOINT_SET_METADATA_KEY] = uuid.uuid4().hex
    checkpoint_metadata[ENVIRONMENT_STEPS_METADATA_KEY] = step
    save_actor_checkpoint(
        policy_path,
        actor_inference_state(learner),
        metadata=checkpoint_metadata,
    )
    save_checkpoint(
        state_path,
        training_checkpoint(learner),
        metadata=_metadata(checkpoint_metadata, "training_state"),
    )
    if replay_path is not None:
        save_checkpoint(
            replay_path,
            learner.replay,
            metadata=_metadata(checkpoint_metadata, "replay"),
        )
    return CheckpointPaths(policy=policy_path, training_state=state_path, replay=replay_path)


def _validate_metadata(path: str | Path, expected_kind: str) -> dict[str, str]:
    metadata = read_checkpoint_metadata(path)
    if metadata.get(ALGORITHM_METADATA_KEY) != ALGORITHM_NAME:
        raise ValueError(
            f"checkpoint algorithm is {metadata.get(ALGORITHM_METADATA_KEY)!r}; expected {ALGORITHM_NAME!r}"
        )
    if metadata.get(CHECKPOINT_KIND_METADATA_KEY) != expected_kind:
        raise ValueError(
            f"checkpoint kind is {metadata.get(CHECKPOINT_KIND_METADATA_KEY)!r}; expected {expected_kind!r}"
        )
    return metadata


def load_actor_checkpoint(
    path: str | Path,
    template: ActorInferenceState,
) -> tuple[ActorInferenceState, dict[str, str]]:
    """Restore a deployable actor after validating algorithm and checkpoint kind."""
    metadata = _validate_metadata(path, "policy")
    state, _ = load_checkpoint(path, template)
    return state, metadata


def restore_learner_checkpoint(
    path: str | Path,
    learner: LearnerState,
    *,
    replay_path: str | Path | None = None,
    load_optimizer: bool = True,
    load_reward_normalizer: bool = True,
) -> tuple[LearnerState, dict[str, str]]:
    """Restore compact state into a freshly reset environment and optional committed replay."""
    metadata = _validate_metadata(path, "training_state")
    restored, _ = load_checkpoint(path, training_checkpoint(learner))
    replay = learner.replay
    if replay_path is not None:
        replay_metadata = _validate_metadata(replay_path, "replay")
        for key in (CHECKPOINT_SET_METADATA_KEY, ENVIRONMENT_STEPS_METADATA_KEY):
            if metadata.get(key) != replay_metadata.get(key):
                raise ValueError(
                    f"training-state and replay checkpoints have different {key}: "
                    f"{metadata.get(key)!r} != {replay_metadata.get(key)!r}"
                )
        replay, _ = load_checkpoint(replay_path, replay)
        replay = _flush_pending(replay)
    if load_reward_normalizer:
        reward_normalizer = restored.reward_normalizer.replace(
            discounted_return=jnp.zeros_like(restored.reward_normalizer.discounted_return)
        )
    else:
        reward_normalizer = learner.reward_normalizer
    if load_optimizer:
        actor = restored.actor
        critic = restored.critic
        temperature = restored.temperature
        loss_scale = restored.loss_scale
        update_credit = restored.update_credit
        global_update_step = restored.global_update_step
    else:
        actor = restored.actor.replace(
            optimizer_state=learner.actor.optimizer_state,
            schedule_step=learner.actor.schedule_step,
        )
        critic = restored.critic.replace(
            optimizer_state=learner.critic.optimizer_state,
            schedule_step=learner.critic.schedule_step,
        )
        temperature = restored.temperature.replace(
            optimizer_state=learner.temperature.optimizer_state,
            schedule_step=learner.temperature.schedule_step,
        )
        loss_scale = learner.loss_scale
        update_credit = learner.update_credit
        global_update_step = learner.global_update_step
    learner = learner.replace(
        actor=actor,
        critic=critic,
        target_critic=restored.target_critic,
        temperature=temperature,
        replay=replay,
        reward_normalizer=reward_normalizer,
        exploration=restored.exploration,
        loss_scale=loss_scale,
        key=restored.key,
        update_credit=update_credit,
        environment_steps=restored.environment_steps,
        global_update_step=global_update_step,
    )
    return learner, metadata
