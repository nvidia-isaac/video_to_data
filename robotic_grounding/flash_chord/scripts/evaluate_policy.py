# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score one supported RL checkpoint over a batch of unassisted full-sequence attempts."""

from __future__ import annotations

import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed, resolve_path
from flash_chord.evaluation.harness import (
    evaluate_checkpoint,
    evaluation_scalars,
    write_evaluation_report,
)
from flash_chord.training.flash_sac.evaluation import EvaluationConfig


def publish_summary(cfg: DictConfig, scalars: dict[str, float], environment_steps: int) -> None:
    """Publish terminal metrics as both sortable summary columns and one plottable history point."""
    if str(cfg.logging.mode) == "disabled":
        return
    import wandb

    run = wandb.init(
        entity=cfg.logging.entity,
        project=cfg.logging.project,
        group=cfg.logging.experiment,
        name=cfg.logging.run_name,
        resume="allow",
    )
    # A resumed run drops history at or below its last logged step, which is the final training
    # step, so the point goes one step past it.
    step = max(environment_steps + 1, int(run.summary.get("_step", -1)) + 1)
    run.log(scalars, step=step)
    run.summary.update(scalars)
    run.finish()


@hydra.main(version_base="1.3", config_path="../src/flash_chord/configs", config_name="evaluate_policy")
def main(cfg: DictConfig) -> None:
    """Rebuild the saved experiment from its checkpoint and score one evaluation cohort."""
    evaluation = instantiate_typed(cfg.evaluation, EvaluationConfig)
    outcome = evaluate_checkpoint(
        resolve_path(evaluation.checkpoint),
        world_count=evaluation.world_count,
        start_frame=evaluation.start_frame,
        reset_mode=evaluation.reset_mode,
        motion_start_frame=evaluation.motion_start_frame,
        motion_end_frame=evaluation.motion_end_frame,
        source_root=evaluation.source_root,
    )
    scalars = evaluation_scalars(outcome)
    print(json.dumps(scalars, indent=2, sort_keys=True), flush=True)
    if evaluation.metrics_output:
        print(f"report: {write_evaluation_report(evaluation.metrics_output, outcome)}", flush=True)
    if evaluation.reset_mode == "explicit":
        try:
            publish_summary(cfg, scalars, outcome.environment_steps)
        except Exception as error:  # noqa: BLE001 - metrics are already computed and written
            print(f"warning: could not publish metrics to W&B: {error}", flush=True)
    elif str(cfg.logging.mode) != "disabled":
        print(
            f"diagnostic reset mode {evaluation.reset_mode!r} is written to JSON but not published to W&B",
            flush=True,
        )


if __name__ == "__main__":
    main()
