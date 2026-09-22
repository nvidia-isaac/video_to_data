# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the reference-tracking policy with native JAX FlashSAC."""

from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import jax
import warp as wp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed, resolve_path, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import (
    build_checkpoint_metadata,
    critic_schema,
    policy_schema,
    resolved_reference_config,
)
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.runner import WandbLogger, metric_definitions, run_training
from flash_chord.training.progress import ConsoleTrainingProgress

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["TZ"] = "PST8PDT,M3.2.0,M11.1.0"
time.tzset()


@hydra.main(version_base="1.3", config_path="../src/flash_chord/configs", config_name="train_flash_sac")
def main(cfg: DictConfig) -> None:
    """Compose one FlashSAC experiment and run its compiled Warp/JAX training loop."""
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    output_dir = Path(HydraConfig.get().runtime.output_dir) if cfg.output_dir is None else resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with wp.ScopedDevice(training.device):
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
            motion_start_frame=int(cfg.task.motion_start_frame),
            motion_end_frame=int(cfg.task.motion_end_frame),
        )
        parquet = setup.parquet_path
        reference = setup.reference
        scene = setup.scene
        env = RLEnv(scene, reference, config=env_config)
        env.reset()
        env.capture_step()
        env.reset()
        jax_env = WarpRLEnv(
            env,
            publish_diagnostics=False,
            publish_step_metrics=False,
        )
        environment_config = resolved_dict(cfg)
        environment_config["task"]["parquet"] = str(parquet)
        environment_config["reference"] = resolved_reference_config(reference, str(parquet))
        checkpoint_metadata = build_checkpoint_metadata(
            environment_config,
            policy_schema(env.action, env.observation_strategy),
            critic_schema(env.action, env.observation_strategy, jax_env.critic_context_names),
        )
        logger = None
        if cfg.logging.mode != "disabled":
            logger = WandbLogger(
                training,
                metric_definitions=metric_definitions(jax_env),
                environment_config=environment_config,
                upload_checkpoints=cfg.logging.upload_checkpoints,
                upload_training_state=cfg.logging.upload_training_state,
                mode=cfg.logging.mode,
                dir=str(output_dir),
            )
        learner = run_training(
            jax_env,
            training,
            logger=logger,
            checkpoint_dir=output_dir,
            log_interval=cfg.logging.log_interval,
            voc_scale_override=cfg.voc_scale_override,
            checkpoint_metadata=checkpoint_metadata,
            progress=ConsoleTrainingProgress(training.total_environment_steps, output_dir),
        )
    print(f"completed {int(jax.device_get(learner.environment_steps))} environment steps; checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
