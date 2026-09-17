#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan successful rollout collection, GR00T training steps, and render parallelism."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunPlan:
    """Deterministic run quantities derived from measured inputs."""

    target_successes: int
    measured_success_rate: float
    collection_safety_factor: float
    planned_collection_envs: int
    episodes: int | None
    frames_per_episode: int | None
    action_horizon: int | None
    usable_samples_per_episode: int | None
    usable_training_samples: int | None
    epochs: float | None
    global_batch_size: int | None
    optimizer_steps: int | None
    requested_eval_envs: int | None
    camera_count: int | None
    max_rendered_cameras: int | None
    planned_eval_envs: int | None


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    """Parse a positive float for argparse."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def success_rate(value: str) -> float:
    """Parse a probability in the open-closed interval (0, 1]."""
    parsed = positive_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("must be <= 1")
    return parsed


def build_plan(args: argparse.Namespace) -> RunPlan:
    """Build a run plan after validating cross-field constraints."""
    planned_collection_envs = math.ceil(
        args.target_successes
        / args.measured_success_rate
        * args.collection_safety_factor
    )
    training_values = (
        args.episodes,
        args.frames_per_episode,
        args.action_horizon,
        args.epochs,
        args.global_batch_size,
    )
    if any(value is not None for value in training_values) and not all(
        value is not None for value in training_values
    ):
        raise ValueError(
            "--episodes, --frames-per-episode, --action-horizon, --epochs, "
            "and --global-batch-size must be supplied together"
        )
    usable_per_episode = None
    usable_training_samples = None
    optimizer_steps = None
    if args.episodes is not None:
        if args.action_horizon > args.frames_per_episode:
            raise ValueError(
                f"action horizon {args.action_horizon} exceeds episode frames "
                f"{args.frames_per_episode}"
            )
        usable_per_episode = args.frames_per_episode - args.action_horizon + 1
        usable_training_samples = args.episodes * usable_per_episode
        optimizer_steps = math.ceil(
            args.epochs * usable_training_samples / args.global_batch_size
        )

    capacity_values = (
        args.requested_eval_envs,
        args.camera_count,
        args.max_rendered_cameras,
    )
    if any(value is not None for value in capacity_values) and not all(
        value is not None for value in capacity_values
    ):
        raise ValueError(
            "--requested-eval-envs, --camera-count, and --max-rendered-cameras must be supplied together"
        )
    planned_eval_envs = None
    if args.requested_eval_envs is not None:
        capacity_envs = args.max_rendered_cameras // args.camera_count
        if capacity_envs < 1:
            raise ValueError(
                "max rendered cameras is smaller than one embodiment camera set"
            )
        planned_eval_envs = min(args.requested_eval_envs, capacity_envs)

    return RunPlan(
        target_successes=args.target_successes,
        measured_success_rate=args.measured_success_rate,
        collection_safety_factor=args.collection_safety_factor,
        planned_collection_envs=planned_collection_envs,
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        action_horizon=args.action_horizon,
        usable_samples_per_episode=usable_per_episode,
        usable_training_samples=usable_training_samples,
        epochs=args.epochs,
        global_batch_size=args.global_batch_size,
        optimizer_steps=optimizer_steps,
        requested_eval_envs=args.requested_eval_envs,
        camera_count=args.camera_count,
        max_rendered_cameras=args.max_rendered_cameras,
        planned_eval_envs=planned_eval_envs,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-successes", type=positive_int, required=True)
    parser.add_argument(
        "--measured-success-rate",
        type=success_rate,
        required=True,
        help="Pilot or prior success rate as a fraction, for example 0.45.",
    )
    parser.add_argument(
        "--collection-safety-factor",
        type=positive_float,
        default=1.1,
    )
    parser.add_argument("--episodes", type=positive_int)
    parser.add_argument("--frames-per-episode", type=positive_int)
    parser.add_argument("--action-horizon", type=positive_int)
    parser.add_argument("--epochs", type=positive_float)
    parser.add_argument("--global-batch-size", type=positive_int)
    parser.add_argument("--requested-eval-envs", type=positive_int)
    parser.add_argument("--camera-count", type=positive_int)
    parser.add_argument("--max-rendered-cameras", type=positive_int)
    return parser.parse_args()


def main() -> None:
    """Print the plan as stable, machine-readable JSON."""
    plan = build_plan(parse_args())
    print(json.dumps(asdict(plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
