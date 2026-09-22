# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training loop, logging seams, curriculum updates, and resume behavior."""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from flash_chord.lifecycle.curriculum import CurriculumStage
from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES
from flash_chord.training.checkpoint import load_checkpoint, read_checkpoint_metadata, save_checkpoint
from flash_chord.training.checkpoint_metadata import validate_resume_reference_config
from flash_chord.training.environment import JAXVectorEnv
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.learner import IterationMetrics, LearnerState, create_learner, train_iteration
from flash_chord.training.progress import TrainingProgress


@runtime_checkable
class CurriculumVectorEnv(JAXVectorEnv, Protocol):
    """Optional vector-env capability for applying one curriculum stage."""

    def set_curriculum(self, stage: CurriculumStage) -> None:
        """Update persistent environment parameters between iterations."""
        ...


class MetricLogger(Protocol):
    """Minimal external metric sink."""

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Write one environment-transition-indexed metric record."""
        ...

    def close(self) -> None:
        """Flush and close the sink."""
        ...


@runtime_checkable
class CheckpointLogger(Protocol):
    """Optional metric-logger capability for publishing saved checkpoints."""

    def log_checkpoint(self, path: Path, *, iteration: int, metadata: Mapping[str, object]) -> None:
        """Publish one successfully saved checkpoint."""
        ...


class WandbLogger:
    """Lazy Weights & Biases metric sink."""

    def __init__(
        self,
        config: TrainingConfig,
        metric_definitions: Mapping[str, str] | None = None,
        environment_config: Mapping[str, object] | None = None,
        upload_checkpoints: bool = True,
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

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self.run.log(dict(metrics), step=step)

    def log_checkpoint(self, path: Path, *, iteration: int, metadata: Mapping[str, object]) -> None:
        if not self._upload_checkpoints:
            return
        artifact = self._wandb.Artifact(
            name=f"run-{self.run.id}-checkpoint",
            type="model",
            metadata={"iteration": iteration, **metadata},
        )
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact, aliases=["latest", f"iteration-{iteration}"])

    def close(self) -> None:
        self.run.finish()


def metric_definitions(env: JAXVectorEnv) -> dict[str, str]:
    """Return the exact aggregation semantics stored with each W&B run."""
    definitions = {
        "reward/total_mean_per_env_step": (
            "Arithmetic mean of total composed reward across world_count * rollout_steps environment steps."
        ),
        "reward/dense_mean_per_env_step": (
            "Total mean reward per environment step minus the weighted termination component."
        ),
        "episode/return_mean_per_completed_episode": "Arithmetic mean return over episodes completed in one rollout.",
        "episode/length_mean_per_completed_episode": (
            "Arithmetic mean control-step length over episodes completed in one rollout."
        ),
        "episode/reference_progress_mean_per_completed_episode": (
            "Mean fraction of remaining reference frames advanced from each episode's sampled reset frame."
        ),
        "termination/reference_end_fraction_per_completed_episode": (
            "Fraction of completed episodes that reached the reference end without a tracking failure."
        ),
        "ppo/value_loss_mean_per_minibatch_update": (
            "Clipped value loss averaged over all PPO epochs and minibatch updates in one iteration."
        ),
        "ppo/kl_mean_per_optimizer_sample": (
            "Behavior-to-current policy KL averaged over samples, epochs, and minibatch updates."
        ),
        "ppo/clip_fraction_mean_per_optimizer_sample": (
            "Fraction of optimizer samples outside the PPO probability-ratio clip interval, averaged over updates."
        ),
        "ppo/explained_variance_per_rollout": (
            "One minus Var(value_target - pre_update_value) / Var(value_target) over all rollout world-steps; "
            "zero when target variance is below 1e-8."
        ),
        "ppo/learning_rate": "Adaptive learning rate after the final minibatch update in the PPO iteration.",
        "ppo/skipped_update_fraction_per_iteration": (
            "Fraction of configured minibatch optimizer updates rejected because their update contained NaN or Inf."
        ),
        "policy/action_mean_abs_mean_per_action": (
            "Mean absolute Gaussian action mean across rollout steps, worlds, and action dimensions."
        ),
        "policy/action_std_mean_per_action": (
            "Arithmetic mean Gaussian action standard deviation across rollout steps, worlds, and dimensions."
        ),
        "throughput/env_steps_per_second": (
            "world_count * rollout_steps divided by wall time for rollout, optimization, and metric synchronization."
        ),
        "curriculum/stage_index": "Zero-based curriculum stage active for this PPO iteration.",
        "curriculum/voc_scale": (
            "Global VOC curriculum target for this PPO iteration; newly reset worlds temporarily apply scale 1."
        ),
    }
    if getattr(env, "reset_to_first_frame_probability", None) is not None:
        definitions["curriculum/reset_to_first_frame_probability"] = (
            "Resolved probability that a sampled reset uses frame zero with the normal settling preamble."
        )
    if getattr(env, "immediate_first_frame_probability", None) is not None:
        definitions["curriculum/immediate_first_frame_probability"] = (
            "Resolved probability that an auto-reset exactly matches immediate frame-zero deployment."
        )
    objective_names = tuple(getattr(env, "objective_term_names", ())) or OBJECTIVE_TERM_NAMES
    for name in objective_names:
        definitions[f"reward_components/{name}_weighted_mean_per_env_step"] = (
            f"Raw mean {name} term multiplied by its active curriculum weight and control timestep."
        )
    for name in tuple(getattr(env, "termination_cause_names", ())):
        definitions[f"termination/{name}_fraction_per_completed_episode"] = (
            f"Fraction of completed episodes assigned exclusively to the {name} tracking-failure cause."
        )
    for name in tuple(getattr(env, "tracking_error_names", ())):
        definitions[f"tracking_error/{name}_mean_env_max_per_step"] = (
            f"For each world-step, maximum {name} across tracked entities; then arithmetic mean over the rollout."
        )
    return definitions


def metrics_to_host(
    metrics: IterationMetrics,
    elapsed_seconds: float,
    sample_count: int,
    stage_index: int | None,
    stage: CurriculumStage | None,
    *,
    frame_dt: float = 1.0,
    objective_term_names: tuple[str, ...] = (),
    termination_cause_names: tuple[str, ...] = (),
    tracking_error_names: tuple[str, ...] = (),
    reset_to_first_frame_probability: float | None = None,
    immediate_first_frame_probability: float | None = None,
) -> dict[str, float]:
    """Synchronize one iteration and produce stable logger keys."""
    rollout = metrics.rollout
    optimization = metrics.optimization
    episode_count = float(np.asarray(rollout.episode_count))
    reference_end_count = float(np.asarray(rollout.reference_end_count))
    episode_return_mean = float(np.asarray(rollout.episode_return_sum)) / max(episode_count, 1.0)
    episode_length_mean = float(np.asarray(rollout.episode_length_sum)) / max(episode_count, 1.0)
    episode_reference_progress_mean = float(np.asarray(rollout.episode_reference_progress_sum)) / max(
        episode_count, 1.0
    )
    output = {
        "reward/total_mean_per_env_step": float(np.asarray(rollout.mean_step_reward)),
        "termination/reference_end_fraction_per_completed_episode": reference_end_count / max(episode_count, 1.0),
        "episode/return_mean_per_completed_episode": episode_return_mean,
        "episode/length_mean_per_completed_episode": episode_length_mean,
        "episode/reference_progress_mean_per_completed_episode": episode_reference_progress_mean,
        "ppo/value_loss_mean_per_minibatch_update": float(np.asarray(optimization.value_loss)),
        "ppo/kl_mean_per_optimizer_sample": float(np.asarray(optimization.kl)),
        "ppo/clip_fraction_mean_per_optimizer_sample": float(np.asarray(optimization.clip_fraction)),
        "ppo/explained_variance_per_rollout": float(np.asarray(optimization.explained_variance)),
        "ppo/learning_rate": float(np.asarray(optimization.learning_rate)),
        "ppo/skipped_update_fraction_per_iteration": float(np.asarray(optimization.skipped_update_fraction)),
        "policy/action_mean_abs_mean_per_action": float(np.asarray(rollout.action_mean_abs_mean)),
        "policy/action_std_mean_per_action": float(np.asarray(rollout.action_std_mean)),
        "throughput/env_steps_per_second": sample_count / elapsed_seconds,
    }
    objective_terms = None if rollout.objective_term_mean is None else np.asarray(rollout.objective_term_mean)
    if objective_terms is not None:
        if objective_terms.shape != (len(objective_term_names),):
            raise ValueError(
                f"objective diagnostics have shape {objective_terms.shape}; expected ({len(objective_term_names)},)"
            )
    if stage is not None and stage_index is not None:
        output["curriculum/stage_index"] = float(stage_index)
        output["curriculum/voc_scale"] = stage.voc_scale
        if reset_to_first_frame_probability is not None:
            output["curriculum/reset_to_first_frame_probability"] = reset_to_first_frame_probability
        if immediate_first_frame_probability is not None:
            output["curriculum/immediate_first_frame_probability"] = immediate_first_frame_probability
        if objective_terms is not None:
            contributions = frame_dt * objective_terms * np.asarray(stage.weights_for(objective_term_names))
            for name, contribution in zip(objective_term_names, contributions, strict=True):
                output[f"reward_components/{name}_weighted_mean_per_env_step"] = float(contribution)
            termination_index = objective_term_names.index("termination")
            termination_contribution = float(contributions[termination_index])
            output["reward/dense_mean_per_env_step"] = (
                output["reward/total_mean_per_env_step"] - termination_contribution
            )
    cause_counts = None if rollout.termination_cause_count is None else np.asarray(rollout.termination_cause_count)
    if cause_counts is not None:
        for name, count in zip(termination_cause_names, cause_counts, strict=True):
            output[f"termination/{name}_fraction_per_completed_episode"] = float(count) / max(episode_count, 1.0)
        classified_count = reference_end_count + float(cause_counts.sum())
        if not np.isclose(classified_count, episode_count):
            raise ValueError(
                f"exclusive outcome counts sum to {classified_count}; expected completed episode count {episode_count}"
            )
    tracking_mean = None if rollout.tracking_error_mean is None else np.asarray(rollout.tracking_error_mean)
    if tracking_mean is not None:
        for name, mean in zip(tracking_error_names, tracking_mean, strict=True):
            output[f"tracking_error/{name}_mean_env_max_per_step"] = float(mean)
    return output


def run_training(
    env: JAXVectorEnv,
    config: TrainingConfig,
    *,
    logger: MetricLogger | None = None,
    checkpoint_dir: str | Path | None = None,
    log_interval: int = 1,
    voc_scale_override: float | None = None,
    checkpoint_metadata: Mapping[str, object] | None = None,
    progress: TrainingProgress | None = None,
) -> LearnerState:
    """Run PPO iterations with optional logging, checkpointing, resume, and VOC override."""
    if log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {log_interval}")
    if voc_scale_override is not None and voc_scale_override < 0.0:
        raise ValueError(f"voc_scale_override must be non-negative, got {voc_scale_override}")

    learner = create_learner(env, config)
    resume_metadata: Mapping[str, str] = {}
    if config.resume:
        resume_metadata = read_checkpoint_metadata(config.checkpoint)
        validate_resume_reference_config(resume_metadata, checkpoint_metadata)
        learner, resume_metadata = load_checkpoint(config.checkpoint, learner)
        learner = learner.replace(observation=jnp.array(env.reset(), copy=True))
    start_iteration = int(np.asarray(learner.iteration))
    sample_count = config.world_count * config.rollout_steps
    environment_steps = int(resume_metadata.get("environment_steps", start_iteration * sample_count))
    checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    if progress is not None:
        progress.start(environment_steps)

    for iteration in range(start_iteration, config.max_iterations):
        stage = None
        stage_index = None
        if config.curriculum is not None:
            if not isinstance(env, CurriculumVectorEnv):
                raise TypeError("configured curriculum requires an environment with set_curriculum(stage)")
            stage_index = config.curriculum.stage_index(iteration)
            stage = config.curriculum.stages[stage_index]
            if voc_scale_override is not None:
                stage = replace(stage, voc_scale=voc_scale_override)
            env.set_curriculum(stage)

        start = time.perf_counter()
        learner, metrics = train_iteration(env, learner, config)
        environment_steps += sample_count
        if progress is not None:
            progress.update(environment_steps)
        if logger is not None and ((iteration + 1) % log_interval == 0 or iteration + 1 == config.max_iterations):
            jax.block_until_ready(metrics.optimization.policy_loss)
            logger.log(
                metrics_to_host(
                    metrics,
                    elapsed_seconds=time.perf_counter() - start,
                    sample_count=sample_count,
                    stage_index=stage_index,
                    stage=stage,
                    frame_dt=float(getattr(env, "frame_dt", 1.0)),
                    objective_term_names=tuple(getattr(env, "objective_term_names", ())),
                    termination_cause_names=tuple(getattr(env, "termination_cause_names", ())),
                    tracking_error_names=tuple(getattr(env, "tracking_error_names", ())),
                    reset_to_first_frame_probability=getattr(env, "reset_to_first_frame_probability", None),
                    immediate_first_frame_probability=getattr(env, "immediate_first_frame_probability", None),
                ),
                step=environment_steps,
            )

        if checkpoint_dir is not None and (
            (iteration + 1) % config.save_interval == 0 or iteration + 1 == config.max_iterations
        ):
            metadata = dict(checkpoint_metadata or {})
            metadata["iteration"] = iteration + 1
            metadata["environment_steps"] = environment_steps
            if stage is not None:
                metadata["voc_scale"] = stage.voc_scale
            checkpoint_path = checkpoint_dir / f"model_{iteration + 1}.safetensors"
            save_checkpoint(checkpoint_path, learner, metadata=metadata)
            if isinstance(logger, CheckpointLogger):
                artifact_metadata = {
                    "iteration": iteration + 1,
                    "environment_steps": environment_steps,
                }
                if stage is not None:
                    artifact_metadata["voc_scale"] = stage.voc_scale
                    reset_probability = getattr(env, "reset_to_first_frame_probability", None)
                    if reset_probability is not None:
                        artifact_metadata["reset_to_first_frame_probability"] = reset_probability
                    immediate_probability = getattr(env, "immediate_first_frame_probability", None)
                    if immediate_probability is not None:
                        artifact_metadata["immediate_first_frame_probability"] = immediate_probability
                logger.log_checkpoint(
                    checkpoint_path,
                    iteration=iteration + 1,
                    metadata=artifact_metadata,
                )

    if logger is not None:
        logger.close()
    return learner
