# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-driven object-tracking evaluation.

Rebuilds one environment from a checkpoint's own resolved config, forces the deterministic
evaluation composition, records a cohort, and scores it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.metrics import (
    ObjectTrackingMetrics,
    TrackingThresholds,
    compute_object_tracking_metrics,
    object_mppe_cm,
)
from flash_chord.evaluation.rollout import ObjectPoseRollout, record_object_pose_rollout
from flash_chord.lifecycle.curriculum import CurriculumStage
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import load_checkpoint_for_inference, read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import (
    TrainingConfig as FlashSACTrainingConfig,
)
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference as load_flash_sac_policy
from flash_chord.training.flash_sac.runner import resolve_jax_device
from flash_chord.training.ppo.config import TrainingConfig as PPOTrainingConfig
from flash_chord.training.ppo.evaluation import EvaluationConfig as PPOEvaluationConfig
from flash_chord.training.ppo.evaluation import policy_action as ppo_policy_action
from flash_chord.training.ppo.learner import create_learner as create_ppo_learner

_FAILURE_TERMS = ("wrist_position", "wrist_orientation", "object_position", "object_orientation")
_RECON_BODY_THRESHOLD_FIELDS = (
    "pelvis_position_threshold",
    "pelvis_orientation_threshold",
    "palm_position_threshold",
    "palm_orientation_threshold",
    "object_position_threshold",
    "object_orientation_threshold",
)
_DISABLED_THRESHOLD = 1.0e6
_FLASH_SAC_TRAINING_TARGET = "flash_chord.training.flash_sac.config.TrainingConfig"
_PPO_TRAINING_TARGET = "flash_chord.training.ppo.config.TrainingConfig"
TrainingAlgorithm = Literal["flash_sac", "ppo"]
EvaluationResetMode = Literal["explicit", "sampled_settled"]


@dataclass(frozen=True, slots=True)
class ReferenceTrackingCriteria:
    """Original failure criteria used to score a rollout after disabling early termination."""

    error_names: tuple[str, ...]
    thresholds: tuple[float | None, ...]
    active_after_step: tuple[int, ...]
    causes: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class ReferenceTrackingMetrics:
    """Whole-reference success under the training task's own failure thresholds."""

    success_rate: float
    reference_end_fraction: float
    failure_cause_fraction: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """One scored cohort and the exact conditions that produced it."""

    metrics: ObjectTrackingMetrics
    reference_tracking: ReferenceTrackingMetrics
    checkpoint: str
    checkpoint_sha256: str
    environment_steps: int
    parquet: str
    world_count: int
    start_frame: int
    step_count: int
    reset_mode: EvaluationResetMode
    settling_steps: int
    motion_start_frame: int
    motion_end_frame: int
    sim_control_fps: float
    sim_physics_fps: float
    object_position_threshold_m: float | None
    object_orientation_threshold_rad: float | None
    non_finite_worlds: tuple[int, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _term_limit(termination: Mapping[str, Any], name: str) -> float | None:
    """Threshold for one termination term, or ``None`` when the term is absent or disabled."""
    term = termination.get(name)
    if term is None or not bool(term.get("enabled", False)):
        return None
    return float(term["threshold"])


def evaluation_thresholds(config: Mapping[str, Any]) -> TrackingThresholds:
    """Map the task's hand/object limits onto the legacy object-tracking metrics."""
    criteria = reference_tracking_criteria(config)
    limits = dict(zip(criteria.error_names, criteria.thresholds, strict=True))
    return TrackingThresholds(
        wrist_position_m=limits.get("wrist_position_m", limits.get("palm_position_m")),
        wrist_orientation_rad=limits.get("wrist_orientation_rad", limits.get("palm_orientation_rad")),
        object_position_m=limits.get("object_position_m"),
        object_orientation_rad=limits.get("object_orientation_rad"),
    )


def _is_recon_body_termination(termination: Mapping[str, Any]) -> bool:
    return all(name in termination for name in _RECON_BODY_THRESHOLD_FIELDS)


def reference_tracking_criteria(config: Mapping[str, Any]) -> ReferenceTrackingCriteria:
    """Recover the task-native threshold semantics before evaluation disables termination."""
    termination = config["env"]["termination"]
    if _is_recon_body_termination(termination):
        freeze = int(termination["reset_freeze_steps"])
        return ReferenceTrackingCriteria(
            error_names=(
                "pelvis_position_m",
                "pelvis_orientation_rad",
                "palm_position_m",
                "palm_orientation_rad",
                "object_position_m",
                "object_orientation_rad",
            ),
            thresholds=tuple(float(termination[name]) for name in _RECON_BODY_THRESHOLD_FIELDS),
            active_after_step=(-1, -1, freeze, freeze, freeze, freeze),
            causes=(("pelvis", (0, 1)), ("palm", (2, 3)), ("object", (4, 5))),
        )

    return ReferenceTrackingCriteria(
        error_names=(
            "wrist_position_m",
            "wrist_orientation_rad",
            "object_position_m",
            "object_orientation_rad",
        ),
        thresholds=tuple(_term_limit(termination, name) for name in _FAILURE_TERMS),
        active_after_step=(-1, -1, -1, -1),
        causes=(("wrist", (0, 1)), ("object", (2, 3))),
    )


def apply_evaluation_composition(
    config: dict[str, Any],
    *,
    world_count: int,
    reset_mode: EvaluationResetMode = "explicit",
) -> dict[str, Any]:
    """Force the unassisted, non-resetting, full-sequence composition onto a checkpoint config.

    Mutates only this in-memory copy. The tracking-failure terms are disabled so a failed world
    keeps advancing instead of freezing at its failure frame; their limits are applied afterwards
    to the recorded errors, and ``reference_end`` completes every world.
    """
    config["scene"]["world_count"] = world_count
    config["env"]["auto_reset"] = False
    config["env"]["voc_scale"] = 0.0
    if reset_mode == "explicit":
        config["env"]["voc"]["max_force"] = 0.0
        config["env"]["voc"]["max_torque"] = 0.0
    config["env"]["reset"]["always_reset_to_first_frame"] = True
    if reset_mode == "explicit":
        config["env"]["reset"]["voc_decay_steps"] = 0
    config["env"]["reset"]["seed"] = 42
    # The resolved config inlines termination under `env` as well as at the top level.
    for termination in _termination_blocks(config):
        if _is_recon_body_termination(termination):
            for name in _RECON_BODY_THRESHOLD_FIELDS:
                termination[name] = _DISABLED_THRESHOLD
            if not bool(termination["truncate_at_reference_end"]):
                raise ValueError("evaluation requires reference-end truncation so every world completes")
        else:
            for name in _FAILURE_TERMS:
                if name in termination:
                    termination[name]["enabled"] = False
            if not bool(termination["reference_end"]["enabled"]):
                raise ValueError("evaluation requires reference-end truncation so every world completes")
    return config


def _shift_criteria_for_preamble(
    criteria: ReferenceTrackingCriteria,
    settling_steps: int,
) -> ReferenceTrackingCriteria:
    """Express task termination activation in scored-step coordinates after a preamble."""
    return replace(
        criteria,
        active_after_step=tuple(max(step - settling_steps, -1) for step in criteria.active_after_step),
    )


def _evaluation_curriculum_stage(
    stage: CurriculumStage,
    reset_mode: EvaluationResetMode,
) -> CurriculumStage:
    """Remove training-only assistance without mixing reset timelines during evaluation."""
    if reset_mode == "sampled_settled":
        return replace(stage, voc_scale=0.0, immediate_first_frame_probability=0.0)
    return replace(stage, voc_scale=0.0)


def _termination_blocks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocks = [config["env"]["termination"]]
    top_level = config.get("termination")
    if isinstance(top_level, dict):
        blocks.append(top_level)
    return blocks


def _resolve_parquet(config: dict[str, Any], source_root: Path | None) -> str:
    saved = Path(config["task"]["parquet"])
    if source_root is not None:
        if saved.is_absolute() and saved.is_relative_to("/workspace"):
            saved = source_root / saved.relative_to("/workspace")
        elif not saved.is_absolute():
            saved = source_root / saved
    return str(saved.resolve())


def checkpoint_training_algorithm(config: Mapping[str, Any]) -> TrainingAlgorithm:
    """Identify a supported learner from its exact typed-config target."""
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("checkpoint resolved config is missing its training mapping")
    target = training.get("_target_")
    if target == _FLASH_SAC_TRAINING_TARGET:
        return "flash_sac"
    if target == _PPO_TRAINING_TARGET:
        return "ppo"
    raise ValueError(f"unsupported checkpoint training target {target!r}")


def _ppo_policy_for_inference(
    checkpoint: Path,
    training: PPOTrainingConfig,
    jax_env: WarpRLEnv,
):
    """Restore PPO policy state against the evaluation world's fresh observation batch."""
    learner, _ = load_checkpoint_for_inference(checkpoint, create_ppo_learner(jax_env, training))
    evaluation = PPOEvaluationConfig(checkpoint=str(checkpoint), deterministic=True)

    def policy_action(learner_state, observation, key):
        return ppo_policy_action(learner_state, observation, training, evaluation), key

    return learner, policy_action, learner.key


def _settle_sampled_reset(
    env: RLEnv,
    jax_env: WarpRLEnv,
    policy_action,
    actor_state,
    key,
    observation,
) -> tuple[Any, Any, int]:
    """Run ReconBody's sampled-reset preamble and return the first unassisted observation."""
    reset_policy = env.reset_policy
    reset_config = getattr(reset_policy, "config", None)
    settling_steps = getattr(reset_config, "reset_freeze_steps", None)
    required = ("reset_frame", "steps_since_reset", "applied_voc_scale")
    if settling_steps is None or any(not hasattr(reset_policy, name) for name in required):
        raise TypeError("sampled_settled evaluation requires a reset policy with a settling timeline")
    settling_steps = int(settling_steps)
    for _ in range(settling_steps):
        action, key = policy_action(actor_state, observation, key)
        observation = jax_env.step(action).observation

    reset_frame = np.asarray(reset_policy.reset_frame.numpy())
    age = np.asarray(reset_policy.steps_since_reset.numpy())
    reference_frame = np.asarray(env.timestep.numpy())
    applied_voc = np.asarray(reset_policy.applied_voc_scale.numpy())
    if np.any(reset_frame != 0):
        raise RuntimeError("sampled-settled evaluation did not reset every world to local frame zero")
    if np.any(age != settling_steps) or np.any(reference_frame != 0):
        raise RuntimeError(
            "sampled-settled preamble did not end at the held first frame: "
            f"age=[{age.min()}, {age.max()}], frame=[{reference_frame.min()}, {reference_frame.max()}]"
        )
    if not np.allclose(applied_voc, 0.0, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            "sampled-settled preamble did not remove VOC before scoring: "
            f"scale=[{applied_voc.min()}, {applied_voc.max()}]"
        )
    return observation, key, settling_steps


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    world_count: int = 4096,
    start_frame: int = 0,
    reset_mode: EvaluationResetMode = "explicit",
    motion_start_frame: int | None = None,
    motion_end_frame: int | None = None,
    source_root: str | Path | None = None,
) -> EvaluationOutcome:
    """Score one actor checkpoint over ``world_count`` unassisted full-sequence attempts."""
    import jax
    import jax.numpy as jnp

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    metadata = read_checkpoint_metadata(checkpoint_path)
    saved = checkpoint_config(metadata)

    algorithm = checkpoint_training_algorithm(saved)
    config = saved
    if reset_mode not in ("explicit", "sampled_settled"):
        raise ValueError(f"unsupported evaluation reset mode {reset_mode!r}")
    if motion_start_frame is not None:
        config["task"]["motion_start_frame"] = motion_start_frame
    if motion_end_frame is not None:
        config["task"]["motion_end_frame"] = motion_end_frame
    thresholds = evaluation_thresholds(config)
    reference_criteria = reference_tracking_criteria(config)
    apply_evaluation_composition(config, world_count=world_count, reset_mode=reset_mode)
    config["task"]["parquet"] = _resolve_parquet(config, Path(source_root) if source_root else None)

    cfg = OmegaConf.create(config)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    if algorithm == "flash_sac":
        training = instantiate_typed(cfg.training, FlashSACTrainingConfig)
    else:
        training = replace(instantiate_typed(cfg.training, PPOTrainingConfig), world_count=world_count)
    device = resolve_jax_device(training.device)

    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=env_config.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=cfg.scene.world_count,
        include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get("object_scale_min", 1.0),
        object_scale_max=cfg.scene.get("object_scale_max", 1.0),
        object_scale_seed=cfg.scene.get("object_scale_seed", 0),
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
    )
    if not 0 <= start_frame < setup.reference.num_frames:
        raise ValueError(f"start_frame must be in [0, {setup.reference.num_frames}), got {start_frame}")
    step_count = setup.reference.num_frames - start_frame

    env = RLEnv(setup.scene, setup.reference, config=env_config)
    env.reset()
    env.capture_step()
    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    if training.curriculum is not None:
        if algorithm == "flash_sac":
            if "curriculum_stage" not in metadata:
                raise ValueError("FlashSAC checkpoint is missing its curriculum stage")
            stage_index = int(metadata["curriculum_stage"])
        else:
            if "iteration" not in metadata:
                raise ValueError("PPO checkpoint is missing its completed iteration")
            stage_index = training.curriculum.stage_index(max(int(metadata["iteration"]) - 1, 0))
        jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[stage_index], reset_mode))

    # PPO template creation resets the environment. Restore the checkpoint before the final,
    # explicit reset so evaluation never inherits ReconBody's sampled-reset settling hold.
    if algorithm == "ppo":
        actor_state, policy_action, key = _ppo_policy_for_inference(
            checkpoint_path,
            training,
            jax_env,
        )

    if reset_mode == "explicit":
        env.reset(frame_id=start_frame)
    else:
        if start_frame != 0:
            raise ValueError("sampled_settled evaluation requires start_frame=0")
        env.reset()
    reset_observation = jnp.asarray(env.observation.numpy().reshape(env.world_count, env.observation_dim))
    observation = jax.device_put(jnp.array(reset_observation, copy=True), device)
    if algorithm == "flash_sac":
        actor_state, policy_action, key, _ = load_flash_sac_policy(
            str(checkpoint_path),
            training,
            observation,
            env.action.action_dim,
            deterministic=True,
            device=device,
        )
    settling_steps = 0
    if reset_mode == "sampled_settled":
        observation, key, settling_steps = _settle_sampled_reset(
            env,
            jax_env,
            policy_action,
            actor_state,
            key,
            observation,
        )
        reference_criteria = _shift_criteria_for_preamble(reference_criteria, settling_steps)
    rollout = record_object_pose_rollout(
        env,
        jax_env,
        policy_action,
        actor_state,
        key,
        observation,
        reference=setup.reference,
        step_count=step_count,
        start_frame=start_frame,
    )
    reference_tracking = score_reference_tracking(rollout, reference_criteria)
    metrics = replace(
        score_rollout(rollout, thresholds),
        chord_sr=reference_tracking.success_rate,
    )
    return EvaluationOutcome(
        metrics=metrics,
        reference_tracking=reference_tracking,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=file_sha256(checkpoint_path),
        environment_steps=int(metadata.get("environment_steps", -1)),
        parquet=str(cfg.task.parquet),
        world_count=env.world_count,
        start_frame=start_frame,
        step_count=step_count,
        reset_mode=reset_mode,
        settling_steps=settling_steps,
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
        sim_control_fps=float(env_config.sim.fps),
        sim_physics_fps=float(env_config.sim.physics_fps),
        object_position_threshold_m=thresholds.object_position_m,
        object_orientation_threshold_rad=thresholds.object_orientation_rad,
        non_finite_worlds=rollout.non_finite_worlds,
    )


def score_rollout(rollout: ObjectPoseRollout, thresholds: TrackingThresholds) -> ObjectTrackingMetrics:
    """Apply the legacy object-centric metrics to either wrist or ReconBody palm diagnostics."""
    names = tuple(rollout.tracking_error_names)
    standard_names = (
        "wrist_position_m" if "wrist_position_m" in names else "palm_position_m",
        "wrist_orientation_rad" if "wrist_orientation_rad" in names else "palm_orientation_rad",
        "object_position_m",
        "object_orientation_rad",
    )
    try:
        error_indices = tuple(names.index(name) for name in standard_names)
    except ValueError as error:
        raise ValueError(f"evaluation diagnostics {names} do not provide {standard_names}") from error
    metrics = compute_object_tracking_metrics(
        rollout.achieved_pose_w,
        rollout.reference_pose_w,
        rollout.object_vertices_o,
        rollout.body_object_ids,
        rollout.tracking_error[..., error_indices],
        thresholds,
        rollout.completion,
        rollout.object_body_names,
    )
    if rollout.non_finite_worlds:
        # The recorder zeroes invalid worlds; those placeholders must not contribute to MPPE.
        valid = np.ones(rollout.achieved_pose_w.shape[1], dtype=np.bool_)
        valid[list(rollout.non_finite_worlds)] = False
        metrics = replace(
            metrics,
            mppe_cm=object_mppe_cm(rollout.achieved_pose_w[:, valid], rollout.reference_pose_w),
        )
    return metrics


def score_reference_tracking(
    rollout: ObjectPoseRollout,
    criteria: ReferenceTrackingCriteria,
) -> ReferenceTrackingMetrics:
    """Score completion without letting an early failure freeze the evaluated trajectory."""
    errors = np.asarray(rollout.tracking_error, dtype=np.float64)
    if errors.ndim != 3 or errors.shape[-1] != len(criteria.error_names):
        raise ValueError(
            f"tracking errors must have shape [step, world, {len(criteria.error_names)}], got {errors.shape}"
        )
    if tuple(rollout.tracking_error_names) != criteria.error_names:
        raise ValueError(
            f"rollout errors {rollout.tracking_error_names} do not match evaluation criteria {criteria.error_names}"
        )
    if len(criteria.thresholds) != errors.shape[-1] or len(criteria.active_after_step) != errors.shape[-1]:
        raise ValueError("reference criteria must provide one threshold and activation step per error")

    worlds = errors.shape[1]
    completion = rollout.completion
    if completion.terminated.shape != (worlds,) or completion.truncated.shape != (worlds,):
        raise ValueError("completion arrays must provide one value per evaluation world")
    if completion.terminated.any():
        raise ValueError("evaluation must disable task failure termination before scoring the full reference")
    reached_end = completion.truncated & np.isclose(completion.reference_progress, 1.0, rtol=0.0, atol=1.0e-6)
    invalid = ~np.isfinite(errors).all(axis=(0, 2))
    if rollout.non_finite_worlds:
        invalid[np.asarray(rollout.non_finite_worlds, dtype=np.int64)] = True

    step = np.arange(errors.shape[0])[:, None]
    cause_masks = []
    for _, indices in criteria.causes:
        failed = np.zeros(worlds, dtype=np.bool_)
        for index in indices:
            threshold = criteria.thresholds[index]
            if threshold is None:
                continue
            active = step > criteria.active_after_step[index]
            failed |= np.any(active & (errors[..., index] > threshold), axis=0)
        cause_masks.append(failed)
    any_failure = np.logical_or.reduce(cause_masks) if cause_masks else np.zeros(worlds, dtype=np.bool_)
    success = reached_end & ~invalid & ~any_failure
    return ReferenceTrackingMetrics(
        success_rate=float(np.mean(success)),
        reference_end_fraction=float(np.mean(reached_end)),
        failure_cause_fraction={
            name: float(np.mean(failed)) for (name, _), failed in zip(criteria.causes, cause_masks, strict=True)
        },
    )


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {name: _json_value(field) for name, field in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(name): _json_value(field) for name, field in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(field) for field in value]
    return value


def write_evaluation_report(path: str | Path, outcome: EvaluationOutcome) -> Path:
    """Atomically write one evaluation report as JSON, without pickle."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_value(outcome), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".json", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        # mkstemp creates 0600, which the OSMO upload worker cannot read.
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def evaluation_scalars(outcome: EvaluationOutcome) -> dict[str, float]:
    """Flat terminal metrics for a run summary."""
    metrics = outcome.metrics
    scalars = {
        "evaluation/chord_sr": metrics.chord_sr,
        "evaluation/add_auc": metrics.add_auc,
        "evaluation/spider_sr_uncentered": metrics.spider_sr_uncentered,
        "evaluation/maniptrans_sr": metrics.maniptrans_sr,
        "evaluation/object_position_error_m": metrics.object_position_error_m,
        "evaluation/object_position_error_std_m": metrics.object_position_error_std_m,
        "evaluation/object_orientation_error_deg": metrics.object_orientation_error_deg,
        "evaluation/object_orientation_error_std_deg": metrics.object_orientation_error_std_deg,
        "evaluation/mean_add_m": metrics.mean_add_m,
        "evaluation/mppe_cm": metrics.mppe_cm,
        "evaluation/add_std_m": metrics.add_std_m,
        "evaluation/spider_position_error_centered_m": metrics.spider_position_error_centered_m,
        "evaluation/world_count": float(outcome.world_count),
        "evaluation/environment_steps": float(outcome.environment_steps),
    }
    scalars.update(
        {f"evaluation/cause_fraction/{name}": value for name, value in metrics.termination_cause_fraction.items()}
    )
    return scalars
