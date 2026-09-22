# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashSAC collection loop, concise metrics, curriculum, checkpointing, and W&B seams."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from flash_chord.lifecycle.curriculum import CurriculumStage
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import (
    checkpoint_critic_schema,
    checkpoint_policy_schema,
    validate_critic_schema,
    validate_policy_schema,
    validate_resume_reference_config,
)
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.checkpoint import (
    CheckpointPaths,
    restore_learner_checkpoint,
    save_learner_checkpoint,
)
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.learner import (
    LearnerExecutables,
    LearnerMetrics,
    LearnerState,
    Transition,
    create_learner,
    reset_context_indices,
)
from flash_chord.training.progress import TrainingProgress

_DEVICE_PATTERN = re.compile(r"^(cuda|gpu|cpu)(?::([0-9]+))?$")


def resolve_jax_device(name: str) -> jax.Device:
    """Resolve one Warp-style device name to the matching logical JAX device."""
    match = _DEVICE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"unsupported JAX device {name!r}; expected cuda:N, gpu:N, or cpu:N")
    requested_platform, raw_index = match.groups()
    platform = "gpu" if requested_platform in {"cuda", "gpu"} else "cpu"
    index = int(raw_index or 0)
    devices = jax.devices(platform)
    if index >= len(devices):
        raise ValueError(f"JAX exposes {len(devices)} {platform} device(s); cannot select {name!r}")
    return devices[index]


def _compile_executables(
    executables: LearnerExecutables,
    learner: LearnerState,
    *,
    objective_weights: jax.Array,
    frame_dt: jax.Array,
    target_voc_scale: jax.Array,
    action_dim: int,
    objective_term_count: int,
    device: jax.Device,
) -> LearnerExecutables:
    """Compile both stable JAX boundaries before timing the first real environment transition."""
    initial_action = executables.initial_action.lower(
        learner.actor.params,
        learner.actor.batch_stats,
        learner.replay.size,
        learner.observation,
        learner.exploration,
        learner.key,
    ).compile()
    world_count = learner.observation.shape[0]
    dummy_transition = Transition(
        action=jax.device_put(np.zeros((world_count, action_dim), dtype=np.float32), device),
        objective_terms=jax.device_put(
            np.zeros((world_count, objective_term_count), dtype=np.float32),
            device,
        ),
        terminated=jax.device_put(np.zeros(world_count, dtype=np.int32), device),
        truncated=jax.device_put(np.zeros(world_count, dtype=np.int32), device),
        episode_reference_progress=jax.device_put(np.zeros(world_count, dtype=np.float32), device),
        termination_diagnostics=jax.device_put(
            np.zeros(
                (
                    world_count,
                    learner.logging.termination_cause_count.size + learner.logging.tracking_error_sum.size,
                ),
                dtype=np.float32,
            ),
            device,
        ),
        next_observation=jnp.zeros_like(learner.observation),
        next_critic_context=jnp.zeros_like(learner.critic_context),
        post_reset_observation=jnp.zeros_like(learner.observation),
        post_reset_critic_context=jnp.zeros_like(learner.critic_context),
    )
    consume_transition = executables.consume_transition.lower(
        learner,
        dummy_transition,
        objective_weights,
        frame_dt,
        target_voc_scale,
        jax.device_put(np.asarray(False, dtype=np.bool_), device),
        jax.device_put(np.asarray(False, dtype=np.bool_), device),
    ).compile()
    return LearnerExecutables(initial_action=initial_action, consume_transition=consume_transition)


def _required_checkpoint_schemas(
    current_metadata: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Require the policy and critic schemas needed to save or resume training safely."""
    if current_metadata is None:
        raise ValueError("FlashSAC checkpointing requires checkpoint metadata with policy and critic schemas")
    current_strings = {key: value for key, value in current_metadata.items() if isinstance(value, str)}
    current_policy = checkpoint_policy_schema(current_strings)
    current_critic = checkpoint_critic_schema(current_strings)
    return current_policy, current_critic


def _validate_resume_schemas(
    saved_metadata: Mapping[str, str],
    current_policy: Mapping[str, object],
    current_critic: Mapping[str, object],
) -> None:
    """Validate saved policy and critic semantics before restoring learner arrays."""
    saved_policy = checkpoint_policy_schema(saved_metadata)
    validate_policy_schema(saved_policy, current_policy)
    saved_critic = checkpoint_critic_schema(saved_metadata)
    validate_critic_schema(saved_critic, current_critic)


class MetricLogger(Protocol):
    """Minimal external metric sink."""

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Write one environment-step-indexed metric record."""
        ...

    def close(self) -> None:
        """Flush and close the sink."""
        ...


@runtime_checkable
class CheckpointLogger(Protocol):
    """Optional logger capability for publishing checkpoint files."""

    def log_checkpoint(
        self,
        paths: CheckpointPaths,
        *,
        environment_steps: int,
        metadata: Mapping[str, object],
    ) -> None:
        """Publish one policy/state/replay checkpoint set."""
        ...


class WandbLogger:
    """Lazy W&B logger with explicit metric definitions and checkpoint artifacts."""

    def __init__(
        self,
        config: TrainingConfig,
        metric_definitions: Mapping[str, str] | None = None,
        environment_config: Mapping[str, object] | None = None,
        upload_checkpoints: bool = True,
        upload_training_state: bool = False,
        **init_options,
    ) -> None:
        import wandb

        run_config = asdict(config)
        run_config["metric_definitions"] = dict(metric_definitions or {})
        run_config["environment"] = dict(environment_config or {})
        self.run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            group=config.experiment_name,
            name=config.run_name,
            config=run_config,
            **init_options,
        )
        self._wandb = wandb
        self._upload_checkpoints = upload_checkpoints
        self._upload_training_state = upload_training_state

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self.run.log(dict(metrics), step=step)

    def log_checkpoint(
        self,
        paths: CheckpointPaths,
        *,
        environment_steps: int,
        metadata: Mapping[str, object],
    ) -> None:
        if not self._upload_checkpoints:
            return
        artifact = self._wandb.Artifact(
            name=f"run-{self.run.id}-checkpoint",
            type="model",
            metadata={"environment_steps": environment_steps, **metadata},
        )
        artifact.add_file(str(paths.policy), name=paths.policy.name)
        if self._upload_training_state:
            artifact.add_file(str(paths.training_state), name=paths.training_state.name)
            if paths.replay is not None:
                artifact.add_file(str(paths.replay), name=paths.replay.name)
        self.run.log_artifact(
            artifact,
            aliases=["latest", f"environment-steps-{environment_steps}"],
        )

    def close(self) -> None:
        self.run.finish()


def metric_definitions(env: WarpRLEnv) -> dict[str, str]:
    """Return exact aggregation semantics stored with each FlashSAC W&B run."""
    definitions = {
        "reward/total_mean_per_env_step": (
            "Arithmetic mean composed reward across every world-step since the previous log."
        ),
        "reward/dense_mean_per_env_step": (
            "Logging-window total mean reward minus the weighted termination component."
        ),
        "episode/return_mean_per_completed_episode": (
            "Mean return for episodes completed since the previous log; zero if none."
        ),
        "episode/length_mean_per_completed_episode": (
            "Mean control-step length for episodes completed since the previous log; zero if none."
        ),
        "episode/reference_progress_mean_per_completed_episode": (
            "Mean normalized reference progress for episodes completed since the previous log; zero if none."
        ),
        "termination/reference_end_fraction_per_completed_episode": (
            "Fraction of logging-window completed episodes ending by reference exhaustion rather than failure."
        ),
        "sac/actor_loss_mean_per_update": "Mean actor objective over updates executed since the previous log.",
        "sac/critic_loss_mean_per_update": (
            "Mean categorical critic cross-entropy over updates executed since the previous log."
        ),
        "sac/temperature_loss_mean_per_update": (
            "Mean direct-temperature objective over delayed updates executed since the previous log."
        ),
        "policy/entropy_mean_per_actor_update": (
            "Mean squashed-policy entropy estimate over actor updates executed since the previous log."
        ),
        "policy/temperature_mean_per_update": (
            "Mean pre-update entropy temperature over delayed updates executed since the previous log."
        ),
        "policy/action_abs_mean_per_env_action": (
            "Mean absolute transformed action over every collected world, step, and action coordinate."
        ),
        "sac/actor_skipped_fraction": "Fraction of actor attempts whose nonfinite gradients skipped Adam.",
        "sac/critic_skipped_fraction": "Fraction of critic attempts whose nonfinite gradients skipped Adam.",
        "replay/size_transitions": "Number of valid committed n-step transitions in replay after this collection.",
        "replay/fill_fraction": "Committed replay size divided by configured capacity.",
        "reward_normalization/scale_denominator": (
            "Current reward divisor: max(discounted-return std, lifetime max absolute return / support bound)."
        ),
        "mixed_precision/loss_scale": "Shared FP16 actor/critic dynamic loss scale after this collection.",
        "throughput/env_steps_per_second": (
            "Environment transitions since the previous log divided by collection/learning wall time."
        ),
        "curriculum/stage_index": "Zero-based environment-transition curriculum stage.",
        "curriculum/voc_scale": "Global target VOC scale active for the latest transition.",
    }
    if env.reset_to_first_frame_probability is not None:
        definitions["curriculum/reset_to_first_frame_probability"] = (
            "Resolved probability that a low-VOC random reference reset is replaced by frame zero. "
            "The environment-level always-reset flag, when enabled, supersedes this probability."
        )
    if getattr(env, "immediate_first_frame_probability", None) is not None:
        definitions["curriculum/immediate_first_frame_probability"] = (
            "Resolved probability that an auto-reset exactly matches immediate frame-zero deployment."
        )
    for name in env.objective_term_names:
        definitions[f"reward_components/{name}_weighted_mean_per_env_step"] = (
            f"Mean weighted {name} contribution over every world-step since the previous log."
        )
    for name in env.termination_cause_names:
        definitions[f"termination/{name}_fraction_per_completed_episode"] = (
            f"Fraction of logging-window completed episodes assigned exclusively to {name}."
        )
    for name in env.tracking_error_names:
        definitions[f"tracking_error/{name}_mean_env_max_per_step"] = (
            f"Mean across logging-window world-steps of the per-world maximum {name}."
        )
    return definitions


def metrics_to_host(
    learner: LearnerMetrics,
    *,
    elapsed_seconds: float,
    environment_steps_since_log: int,
    stage_index: int,
    stage: CurriculumStage,
    replay_capacity: int,
    objective_term_names: tuple[str, ...],
    termination_cause_names: tuple[str, ...],
    tracking_error_names: tuple[str, ...],
    reset_to_first_frame_probability: float | None = None,
    immediate_first_frame_probability: float | None = None,
) -> dict[str, float]:
    """Synchronize one log boundary and produce concise, explicitly aggregated values."""
    episode_count = float(np.asarray(learner.episode_count))
    denominator = max(episode_count, 1.0)
    reference_end_count = float(np.asarray(learner.reference_end_count))
    replay_size = float(np.asarray(learner.replay_size))
    output = {
        "reward/total_mean_per_env_step": float(np.asarray(learner.reward_mean_per_env_step)),
        "episode/return_mean_per_completed_episode": float(
            np.asarray(learner.episode_return_mean_per_completed_episode)
        ),
        "episode/length_mean_per_completed_episode": float(
            np.asarray(learner.episode_length_mean_per_completed_episode)
        ),
        "episode/reference_progress_mean_per_completed_episode": float(
            np.asarray(learner.episode_reference_progress_mean_per_completed_episode)
        ),
        "termination/reference_end_fraction_per_completed_episode": reference_end_count / denominator,
        "sac/actor_loss_mean_per_update": float(np.asarray(learner.actor_loss_mean_per_update)),
        "sac/critic_loss_mean_per_update": float(np.asarray(learner.critic_loss_mean_per_update)),
        "sac/temperature_loss_mean_per_update": float(np.asarray(learner.temperature_loss_mean_per_update)),
        "policy/entropy_mean_per_actor_update": float(np.asarray(learner.actor_entropy_mean_per_update)),
        "policy/temperature_mean_per_update": float(np.asarray(learner.temperature_value_mean_per_update)),
        "policy/action_abs_mean_per_env_action": float(np.asarray(learner.action_abs_mean_per_env_action)),
        "sac/actor_skipped_fraction": float(np.asarray(learner.actor_skipped_fraction)),
        "sac/critic_skipped_fraction": float(np.asarray(learner.critic_skipped_fraction)),
        "replay/size_transitions": replay_size,
        "replay/fill_fraction": replay_size / replay_capacity,
        "reward_normalization/scale_denominator": float(np.asarray(learner.reward_scale_denominator)),
        "mixed_precision/loss_scale": float(np.asarray(learner.loss_scale)),
        "throughput/env_steps_per_second": environment_steps_since_log / elapsed_seconds,
        "curriculum/stage_index": float(stage_index),
        "curriculum/voc_scale": stage.voc_scale,
    }
    if reset_to_first_frame_probability is not None:
        output["curriculum/reset_to_first_frame_probability"] = reset_to_first_frame_probability
    if immediate_first_frame_probability is not None:
        output["curriculum/immediate_first_frame_probability"] = immediate_first_frame_probability

    contributions = np.asarray(learner.objective_contribution_mean_per_env_step)
    if contributions.shape != (len(objective_term_names),):
        raise ValueError(
            f"objective contributions have shape {contributions.shape}; expected {(len(objective_term_names),)}"
        )
    for name, contribution in zip(objective_term_names, contributions, strict=True):
        output[f"reward_components/{name}_weighted_mean_per_env_step"] = float(contribution)
    if "termination" in objective_term_names:
        termination_index = objective_term_names.index("termination")
        output["reward/dense_mean_per_env_step"] = output["reward/total_mean_per_env_step"] - float(
            contributions[termination_index]
        )

    cause_counts = np.asarray(learner.termination_cause_count)
    if cause_counts.shape != (len(termination_cause_names),):
        raise ValueError(
            f"termination cause counts have shape {cause_counts.shape}; expected {(len(termination_cause_names),)}"
        )
    for name, count in zip(termination_cause_names, cause_counts, strict=True):
        output[f"termination/{name}_fraction_per_completed_episode"] = float(count) / denominator
    classified_count = reference_end_count + float(cause_counts.sum())
    if not np.isclose(classified_count, episode_count):
        raise ValueError(
            f"exclusive outcome counts sum to {classified_count}; expected completed episode count {episode_count}"
        )

    tracking_means = np.asarray(learner.tracking_error_mean_per_env_step)
    if tracking_means.shape != (len(tracking_error_names),):
        raise ValueError(
            f"tracking-error means have shape {tracking_means.shape}; expected {(len(tracking_error_names),)}"
        )
    for name, mean in zip(tracking_error_names, tracking_means, strict=True):
        output[f"tracking_error/{name}_mean_env_max_per_step"] = float(mean)

    return output


def _fallback_stage(env: WarpRLEnv) -> CurriculumStage:
    objective = getattr(env.env, "objective", None)
    weights = getattr(objective, "weights", None)
    voc_scale = getattr(env.env, "voc_scale", None)
    if weights is None or voc_scale is None:
        raise TypeError("training without curriculum requires objective weights and a VOC scale buffer")
    values = np.asarray(weights.numpy())
    if values.shape != (len(env.objective_term_names),):
        raise ValueError(f"objective weights have shape {values.shape}; expected {(len(env.objective_term_names),)}")
    return CurriculumStage(
        voc_scale=float(np.asarray(voc_scale.numpy())[0]),
        objective_weights=dict(zip(env.objective_term_names, values.tolist(), strict=True)),
    )


def _run_training(
    env: WarpRLEnv,
    config: TrainingConfig,
    *,
    logger: MetricLogger | None = None,
    checkpoint_dir: str | Path | None = None,
    log_interval: int = 10,
    voc_scale_override: float | None = None,
    checkpoint_metadata: Mapping[str, object] | None = None,
    progress: TrainingProgress | None = None,
) -> LearnerState:
    """Run one Warp interaction followed by one donated JAX state transition until the step budget is met."""
    if log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {log_interval}")
    if voc_scale_override is not None and voc_scale_override < 0.0:
        raise ValueError(f"voc_scale_override must be non-negative, got {voc_scale_override}")
    if env.world_count != config.world_count:
        raise ValueError(f"environment has {env.world_count} worlds; config requests {config.world_count}")
    context_names = tuple(env.critic_context_names)
    reset_context_indices(context_names)
    if not env.objective_term_names:
        raise ValueError("FlashSAC requires named raw objective-term diagnostics")
    checkpoint_schemas = None
    if config.resume or checkpoint_dir is not None:
        checkpoint_schemas = _required_checkpoint_schemas(checkpoint_metadata)

    fallback = _fallback_stage(env) if config.curriculum is None else None

    def stage_at(environment_steps: int) -> tuple[int, CurriculumStage]:
        if config.curriculum is None:
            if fallback is None:
                raise AssertionError("fallback curriculum stage was not initialized")
            index, stage = 0, fallback
        else:
            index = config.curriculum.stage_index(environment_steps)
            stage = config.curriculum.stages[index]
        if voc_scale_override is not None:
            stage = replace(stage, voc_scale=voc_scale_override)
        return index, stage

    jax_device = resolve_jax_device(config.device)
    _, initial_stage = stage_at(0)
    env.set_curriculum(initial_stage)
    initial_observation = jax.device_put(jnp.array(env.reset(), copy=True), jax_device)
    initial_context = jax.device_put(jnp.array(env.initial_critic_context, copy=True), jax_device)
    learner, executables = create_learner(
        config,
        initial_observation,
        initial_context,
        action_dim=env.action_dim,
        objective_term_count=len(env.objective_term_names),
        termination_cause_count=len(env.termination_cause_names),
        tracking_error_count=len(env.tracking_error_names),
        critic_context_names=context_names,
        device=jax_device,
    )
    if config.resume:
        if checkpoint_schemas is None:
            raise AssertionError("resume schema validation was not initialized")
        saved_metadata = read_checkpoint_metadata(config.checkpoint_path)
        _validate_resume_schemas(saved_metadata, *checkpoint_schemas)
        validate_resume_reference_config(saved_metadata, checkpoint_metadata)
        learner, _ = restore_learner_checkpoint(
            config.checkpoint_path,
            learner,
            replay_path=config.checkpoint.load_replay_path,
            load_optimizer=config.checkpoint.load_optimizer,
            load_reward_normalizer=config.checkpoint.load_reward_normalizer,
        )
        resumed_steps = int(jax.device_get(learner.environment_steps))
        _, initial_stage = stage_at(resumed_steps)
        env.set_curriculum(initial_stage)
        learner = learner.replace(
            observation=learner.observation.at[:].set(jax.device_put(env.reset(), jax_device)),
            critic_context=learner.critic_context.at[:].set(jax.device_put(env.initial_critic_context, jax_device)),
        )

    environment_steps = int(jax.device_get(learner.environment_steps))
    if environment_steps > config.total_environment_steps:
        raise ValueError(
            f"checkpoint has {environment_steps} environment steps; budget is {config.total_environment_steps}"
        )
    if (config.total_environment_steps - environment_steps) % config.world_count != 0:
        raise ValueError("remaining environment-step budget must be divisible by world_count")
    if progress is not None:
        progress.start(environment_steps)

    current_stage_index, current_stage = stage_at(environment_steps)
    env.set_curriculum(current_stage)
    objective_weights = jax.device_put(
        np.asarray(current_stage.weights_for(env.objective_term_names), dtype=np.float32),
        jax_device,
    )
    target_voc_scale = jax.device_put(np.asarray(current_stage.voc_scale, dtype=np.float32), jax_device)
    frame_dt = jax.device_put(np.asarray(env.frame_dt, dtype=np.float32), jax_device)
    stage_changed_false = jax.device_put(np.asarray(False, dtype=np.bool_), jax_device)
    stage_changed_true = jax.device_put(np.asarray(True, dtype=np.bool_), jax_device)
    reset_logging_false = jax.device_put(np.asarray(False, dtype=np.bool_), jax_device)
    reset_logging_true = jax.device_put(np.asarray(True, dtype=np.bool_), jax_device)
    compile_start = time.perf_counter()
    executables = _compile_executables(
        executables,
        learner,
        objective_weights=objective_weights,
        frame_dt=frame_dt,
        target_voc_scale=target_voc_scale,
        action_dim=env.action_dim,
        objective_term_count=len(env.objective_term_names),
        device=jax_device,
    )
    print(f"compiled FlashSAC JAX executables on {jax_device} in {time.perf_counter() - compile_start:.2f}s")
    action, exploration, key = executables.initial_action(
        learner.actor.params,
        learner.actor.batch_stats,
        learner.replay.size,
        learner.observation,
        learner.exploration,
        learner.key,
    )
    learner = learner.replace(exploration=exploration, key=key)

    checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    checkpoint_interval = config.checkpoint.save_interval_environment_steps
    next_checkpoint = ((environment_steps // checkpoint_interval) + 1) * checkpoint_interval
    last_checkpoint_step = -1
    collection_index = 0
    log_start = time.perf_counter()
    steps_since_log = 0

    while environment_steps < config.total_environment_steps:
        next_stage_index, next_stage = stage_at(environment_steps)
        stage_changed = next_stage_index != current_stage_index
        if stage_changed:
            jax.block_until_ready(learner)
            current_stage_index = next_stage_index
            current_stage = next_stage
            env.set_curriculum(current_stage)
            objective_weights = jax.device_put(
                np.asarray(current_stage.weights_for(env.objective_term_names), dtype=np.float32),
                jax_device,
            )
            target_voc_scale = jax.device_put(
                np.asarray(current_stage.voc_scale, dtype=np.float32),
                jax_device,
            )

        next_collection_index = collection_index + 1
        next_environment_steps = environment_steps + config.world_count
        logging_boundary = (
            next_collection_index % log_interval == 0 or next_environment_steps == config.total_environment_steps
        )
        step = env.step(action)
        if step.objective_terms is None:
            raise RuntimeError("environment stopped publishing raw objective terms")
        if step.terminal_observation is None or step.critic_context is None or step.terminal_critic_context is None:
            raise RuntimeError("environment stopped publishing terminal-state outputs")
        episode_reference_progress, termination_diagnostics = env.transition_logging_inputs()
        learner, action, learner_metrics = executables.consume_transition(
            learner,
            Transition(
                action=action,
                objective_terms=step.objective_terms,
                terminated=step.terminated,
                truncated=step.truncated,
                episode_reference_progress=episode_reference_progress,
                termination_diagnostics=termination_diagnostics,
                next_observation=step.terminal_observation,
                next_critic_context=step.terminal_critic_context,
                post_reset_observation=step.observation,
                post_reset_critic_context=step.critic_context,
            ),
            objective_weights,
            frame_dt,
            target_voc_scale,
            stage_changed_true if stage_changed else stage_changed_false,
            reset_logging_true if logging_boundary else reset_logging_false,
        )
        environment_steps = next_environment_steps
        steps_since_log += config.world_count
        collection_index = next_collection_index
        if progress is not None:
            progress.update(environment_steps)

        if logger is not None and logging_boundary:
            jax.block_until_ready(learner_metrics.critic_loss_mean_per_update)
            logger.log(
                metrics_to_host(
                    learner_metrics,
                    elapsed_seconds=time.perf_counter() - log_start,
                    environment_steps_since_log=steps_since_log,
                    stage_index=current_stage_index,
                    stage=current_stage,
                    replay_capacity=config.replay.capacity,
                    objective_term_names=env.objective_term_names,
                    termination_cause_names=env.termination_cause_names,
                    tracking_error_names=env.tracking_error_names,
                    reset_to_first_frame_probability=env.reset_to_first_frame_probability,
                    immediate_first_frame_probability=getattr(env, "immediate_first_frame_probability", None),
                ),
                step=environment_steps,
            )
            log_start = time.perf_counter()
            steps_since_log = 0

        should_save = checkpoint_dir is not None and (
            environment_steps >= next_checkpoint or environment_steps == config.total_environment_steps
        )
        if should_save and environment_steps != last_checkpoint_step:
            jax.block_until_ready(learner.global_update_step)
            checkpoint_start = time.perf_counter()
            curriculum_metadata = {
                "curriculum_stage": current_stage_index,
                "voc_scale": current_stage.voc_scale,
            }
            if env.reset_to_first_frame_probability is not None:
                curriculum_metadata["reset_to_first_frame_probability"] = env.reset_to_first_frame_probability
            immediate_probability = getattr(env, "immediate_first_frame_probability", None)
            if immediate_probability is not None:
                curriculum_metadata["immediate_first_frame_probability"] = immediate_probability
            metadata = dict(checkpoint_metadata or {})
            metadata.update(
                {
                    "environment_steps": environment_steps,
                    "global_update_step": int(jax.device_get(learner.global_update_step)),
                    "replay_included": config.checkpoint.save_replay,
                    **curriculum_metadata,
                }
            )
            paths = save_learner_checkpoint(
                checkpoint_dir,
                learner,
                metadata=metadata,
                save_replay=config.checkpoint.save_replay,
            )
            if isinstance(logger, CheckpointLogger):
                logger.log_checkpoint(
                    paths,
                    environment_steps=environment_steps,
                    metadata=curriculum_metadata,
                )
            last_checkpoint_step = environment_steps
            while next_checkpoint <= environment_steps:
                next_checkpoint += checkpoint_interval
            log_start += time.perf_counter() - checkpoint_start

    return learner


def run_training(
    env: WarpRLEnv,
    config: TrainingConfig,
    *,
    logger: MetricLogger | None = None,
    checkpoint_dir: str | Path | None = None,
    log_interval: int = 10,
    voc_scale_override: float | None = None,
    checkpoint_metadata: Mapping[str, object] | None = None,
    progress: TrainingProgress | None = None,
) -> LearnerState:
    """Run FlashSAC and always close an attached external metric sink."""
    try:
        return _run_training(
            env,
            config,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
            voc_scale_override=voc_scale_override,
            checkpoint_metadata=checkpoint_metadata,
            progress=progress,
        )
    finally:
        if logger is not None:
            logger.close()
