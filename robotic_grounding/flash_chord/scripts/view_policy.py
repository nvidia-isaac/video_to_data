# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""View a PPO or FlashSAC checkpoint against its saved reference motion."""

from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed, resolve_path
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import checkpoint_training_algorithm
from flash_chord.evaluation.view import PolicyViewConfig
from flash_chord.runtime.jax import to_jax
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import load_checkpoint_for_inference, read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import (
    checkpoint_config,
    checkpoint_evaluation_config,
    checkpoint_policy_schema,
    policy_schema,
    validate_policy_schema,
    validate_reference_config,
)
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig as FlashSACTrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference as load_flash_sac_policy
from flash_chord.training.flash_sac.runner import resolve_jax_device
from flash_chord.training.ppo.config import TrainingConfig as PPOTrainingConfig
from flash_chord.training.ppo.evaluation import EvaluationConfig as PPOEvaluationConfig
from flash_chord.training.ppo.evaluation import policy_action as ppo_policy_action
from flash_chord.training.ppo.learner import create_learner
from flash_chord.visualization.markers import log_configured_markers
from flash_chord.visualization.viewer import ViewerApp, ViewerConfig


def _reset_completed_worlds(env: RLEnv, evaluation: PolicyViewConfig):
    """Apply explicit viewer resets without sampling training recovery starts."""
    if evaluation.reset_mode != "explicit":
        return None, False
    completed = np.maximum(env.done.numpy(), env.truncation.numpy())
    if not completed.any():
        return None, False
    reset_observation = env.reset(
        frame_id=evaluation.start_frame,
        reset_mask=wp.array(completed, dtype=wp.int32, device=env.device),
    )
    observation = to_jax(reset_observation, (env.world_count, env.observation_dim))
    return observation, bool(completed[0])


class PPOPolicyViewer(ViewerApp):
    """Viewer app that advances one restored PPO learner."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        cfg = args.config
        env_config = instantiate_typed(cfg.env, RLEnvConfig)
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        collision = instantiate_typed(cfg.collision, CollisionPolicy)
        training = instantiate_typed(cfg.training, PPOTrainingConfig)
        evaluation = instantiate_typed(cfg.evaluation, PolicyViewConfig)
        if evaluation.reset_mode == "explicit":
            # Auto-reset is captured into the environment graph, so disable it before construction.
            env_config = replace(env_config, auto_reset=False)
        policy_evaluation = PPOEvaluationConfig(
            checkpoint=evaluation.checkpoint,
            deterministic=evaluation.deterministic,
            use_checkpoint_config=evaluation.use_checkpoint_config,
            validate_policy_schema=evaluation.validate_policy_schema,
        )
        checkpoint = resolve_path(evaluation.checkpoint)
        device = resolve_jax_device(training.device)
        motion_start_frame = (
            int(cfg.task.motion_start_frame) if evaluation.motion_start_frame is None else evaluation.motion_start_frame
        )
        motion_end_frame = (
            int(cfg.task.motion_end_frame) if evaluation.motion_end_frame is None else evaluation.motion_end_frame
        )

        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=motion_start_frame,
            motion_end_frame=motion_end_frame,
        )
        reference = setup.reference
        scene = setup.scene
        env = RLEnv(scene, reference, config=env_config)
        env.reset()
        env.capture_step()
        env.reset()
        jax_env = WarpRLEnv(env)
        checkpoint_metadata = read_checkpoint_metadata(checkpoint)
        saved_config = checkpoint_config(checkpoint_metadata)
        saved_schema = checkpoint_policy_schema(checkpoint_metadata)
        if evaluation.validate_policy_schema:
            validate_reference_config(saved_config, reference)
            validate_policy_schema(saved_schema, policy_schema(env.action, env.observation_strategy))
        learner, _ = load_checkpoint_for_inference(checkpoint, create_learner(jax_env, training))

        stage_index = None
        if training.curriculum is not None:
            completed_iteration = max(int(learner.iteration) - 1, 0)
            stage_index = training.curriculum.stage_index(completed_iteration)
            stage = training.curriculum.stages[stage_index]
            stage = replace(stage, voc_scale=env_config.voc_scale)
            jax_env.set_curriculum(stage)

        if evaluation.reset_mode == "explicit":
            reset_observation = env.reset(frame_id=evaluation.start_frame)
            observation = jax.device_put(
                jnp.array(
                    to_jax(reset_observation, (env.world_count, env.observation_dim)),
                    copy=True,
                ),
                device,
            )
        else:
            observation = jax.device_put(jnp.array(jax_env.reset(), copy=True), device)

        self.env = env
        self.scene = scene
        self.reference = reference
        self.jax_env = jax_env
        self.learner = learner
        self.training = training
        self.evaluation = evaluation
        self.policy_evaluation = policy_evaluation
        self.observation = observation
        self.key = learner.key
        self.model = scene.model
        self.state = env.state_0
        self._shown_frame = int(env.timestep.numpy()[0])
        self.viewer.set_model(self.model)
        print(
            f"policy view: algorithm=PPO, checkpoint={checkpoint}, iteration={int(learner.iteration)}, "
            f"worlds={env.world_count}, stage={stage_index}, "
            f"policy={'mean' if evaluation.deterministic else 'sample'}, "
            f"voc_target={float(env.voc_scale.numpy()[0]):g}, "
            f"reset_assist_steps={env.reset_policy.config.voc_decay_steps}, "
            f"control={env_config.sim.fps:g} Hz, physics={env_config.sim.physics_fps:g} Hz"
        )

    def step(self) -> None:
        action_key = None
        if not self.evaluation.deterministic:
            self.key, action_key = jax.random.split(self.key)
        action = ppo_policy_action(
            self.learner,
            self.observation,
            self.training,
            self.policy_evaluation,
            action_key,
        )
        action_frame = int(self.env.timestep.numpy()[0])
        transition = self.jax_env.step(action)
        reset_observation, explicit_reset = _reset_completed_worlds(self.env, self.evaluation)
        observation = transition.observation if reset_observation is None else reset_observation
        self.observation = jnp.array(observation, copy=True)
        self.state = self.env.state_0
        reset = explicit_reset or bool(self.env.reset_mask.numpy()[0])
        self._shown_frame = int(self.env.timestep.numpy()[0]) if reset else action_frame
        self.sim_time += self.env.frame_dt

    def render(self) -> None:
        self.pace_realtime()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        log_configured_markers(
            self.viewer,
            self.args.config.markers,
            self.scene,
            self.reference,
            self._shown_frame,
            self.state,
            self.env.contacts,
            self.model,
            self.env.device,
        )
        self.viewer.end_frame()


class FlashSACPolicyViewer(ViewerApp):
    """Viewer app that advances one precompiled FlashSAC actor and captured Warp environment."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        cfg = args.config
        env_config = instantiate_typed(cfg.env, RLEnvConfig)
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        collision = instantiate_typed(cfg.collision, CollisionPolicy)
        training = instantiate_typed(cfg.training, FlashSACTrainingConfig)
        evaluation = instantiate_typed(cfg.evaluation, PolicyViewConfig)
        if evaluation.reset_mode == "explicit":
            env_config = replace(env_config, auto_reset=False)
        checkpoint = resolve_path(evaluation.checkpoint)
        device = resolve_jax_device(training.device)
        motion_start_frame = (
            int(cfg.task.motion_start_frame) if evaluation.motion_start_frame is None else evaluation.motion_start_frame
        )
        motion_end_frame = (
            int(cfg.task.motion_end_frame) if evaluation.motion_end_frame is None else evaluation.motion_end_frame
        )

        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=motion_start_frame,
            motion_end_frame=motion_end_frame,
        )
        reference = setup.reference
        scene = setup.scene
        env = RLEnv(scene, reference, config=env_config)
        env.reset()
        env.capture_step()
        jax_env = WarpRLEnv(
            env,
            publish_diagnostics=False,
            publish_step_metrics=False,
        )
        checkpoint_metadata = read_checkpoint_metadata(checkpoint)
        saved_config = checkpoint_config(checkpoint_metadata)
        saved_schema = checkpoint_policy_schema(checkpoint_metadata)
        if evaluation.validate_policy_schema:
            validate_reference_config(saved_config, reference)
            validate_policy_schema(saved_schema, policy_schema(env.action, env.observation_strategy))

        stage_index = None
        if training.curriculum is not None:
            if "curriculum_stage" not in checkpoint_metadata:
                raise ValueError("FlashSAC policy checkpoint is missing its curriculum stage")
            stage_index = int(checkpoint_metadata["curriculum_stage"])
            if not 0 <= stage_index < len(training.curriculum.stages):
                raise ValueError(
                    f"checkpoint curriculum stage {stage_index} is outside [0, {len(training.curriculum.stages)})"
                )
            stage = replace(training.curriculum.stages[stage_index], voc_scale=env_config.voc_scale)
            jax_env.set_curriculum(stage)

        if evaluation.reset_mode == "explicit":
            reset_observation = env.reset(frame_id=evaluation.start_frame)
            observation = jax.device_put(
                jnp.array(
                    to_jax(reset_observation, (env.world_count, env.observation_dim)),
                    copy=True,
                ),
                device,
            )
        else:
            observation = jax.device_put(jnp.array(jax_env.reset(), copy=True), device)
        actor_state, policy_action, key, _ = load_flash_sac_policy(
            str(checkpoint),
            training,
            observation,
            env.action.action_dim,
            deterministic=evaluation.deterministic,
            device=device,
        )

        self.env = env
        self.scene = scene
        self.reference = reference
        self.jax_env = jax_env
        self.actor_state = actor_state
        self.policy_action = policy_action
        self.evaluation = evaluation
        self.observation = observation
        self.key = key
        self.model = scene.model
        self.state = env.state_0
        self._frame_status = wp.zeros(2, dtype=wp.int32, device=env.device)
        self._next_action_frame = int(env.timestep.numpy()[0])
        self._shown_frame = self._next_action_frame
        self.viewer.set_model(self.model)
        print(
            f"policy view: algorithm=FlashSAC, checkpoint={checkpoint}, "
            f"environment_steps={checkpoint_metadata.get('environment_steps', 'unknown')}, "
            f"worlds={env.world_count}, stage={stage_index}, "
            f"policy={'mean' if evaluation.deterministic else 'sample'}, "
            f"voc_target={float(env.voc_scale.numpy()[0]):g}, "
            f"reset_assist_steps={env.reset_policy.config.voc_decay_steps}, "
            f"control={env_config.sim.fps:g} Hz, physics={env_config.sim.physics_fps:g} Hz"
        )

    def step(self) -> None:
        action, self.key = self.policy_action(
            self.actor_state,
            self.observation,
            self.key,
        )
        action_frame = self._next_action_frame
        transition = self.jax_env.step(action)
        reset_observation, explicit_reset = _reset_completed_worlds(self.env, self.evaluation)
        self.observation = (
            transition.observation if reset_observation is None else jnp.array(reset_observation, copy=True)
        )
        self.state = self.env.state_0
        wp.copy(self._frame_status, self.env.timestep, dest_offset=0, count=1)
        wp.copy(self._frame_status, self.env.reset_mask, dest_offset=1, count=1)
        frame_status = self._frame_status.numpy()
        timestep = int(frame_status[0])
        self._shown_frame = timestep if explicit_reset or frame_status[1] else action_frame
        self._next_action_frame = timestep
        self.sim_time += self.env.frame_dt

    def render(self) -> None:
        self.pace_realtime()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        log_configured_markers(
            self.viewer,
            self.args.config.markers,
            self.scene,
            self.reference,
            self._shown_frame,
            self.state,
            self.env.contacts,
            self.model,
            self.env.device,
        )
        self.viewer.end_frame()


@hydra.main(version_base="1.3", config_path="../src/flash_chord/configs", config_name="view_policy")
def main(cfg: DictConfig) -> None:
    """Detect the checkpoint learner and launch its configured viewer (Viser by default)."""
    requested_evaluation = instantiate_typed(cfg.evaluation, PolicyViewConfig)
    checkpoint = resolve_path(requested_evaluation.checkpoint)
    metadata = read_checkpoint_metadata(checkpoint)
    saved_config = checkpoint_config(metadata)
    algorithm = checkpoint_training_algorithm(saved_config)
    if requested_evaluation.use_checkpoint_config:
        parser = OverridesParser.create()
        overrides = parser.parse_overrides(HydraConfig.get().overrides.task)
        override_keys = tuple(override.get_key_element() for override in overrides)
        cfg = checkpoint_evaluation_config(
            cfg,
            saved_config,
            override_keys,
            use_evaluation_training_world_count=algorithm == "ppo",
        )
        print("evaluation config: saved checkpoint metadata with explicit CLI overrides")
    else:
        print("evaluation config: current Hydra composition (checkpoint config disabled)")
    viewer = instantiate_typed(cfg.viewer, ViewerConfig)
    viewer_app = PPOPolicyViewer if algorithm == "ppo" else FlashSACPolicyViewer
    print(f"checkpoint learner: {algorithm}")
    viewer_app.launch_configured(cfg, viewer, device=cfg.training.device)


if __name__ == "__main__":
    main()
