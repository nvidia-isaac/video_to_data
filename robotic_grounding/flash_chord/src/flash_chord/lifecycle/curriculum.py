# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed curricula for VOC assist and objective weights.

The scale steps down with a monotonic curriculum-step counter (workflow-defined — RL steps or MPPI
optimization iterations).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VOCCurriculum:
    """Fixed-schedule VOC scale annealing keyed on a monotonic curriculum step.

    ``schedule`` is ``(step_threshold, voc_scale)`` pairs with strictly increasing thresholds; the active
    scale is the value of the highest threshold the step has reached, or ``initial_scale`` before the
    first threshold.
    """

    schedule: tuple[tuple[int, float], ...]
    initial_scale: float = 1.0

    def __post_init__(self) -> None:
        thresholds = [s for s, _ in self.schedule]
        if (
            any(threshold < 0 for threshold in thresholds)
            or thresholds != sorted(thresholds)
            or len(set(thresholds)) != len(thresholds)
        ):
            raise ValueError(f"schedule thresholds must be non-negative and strictly increasing, got {thresholds}")
        scales = [self.initial_scale, *(scale for _, scale in self.schedule)]
        if any(not math.isfinite(scale) or scale < 0.0 for scale in scales):
            raise ValueError(f"VOC scales must be finite and non-negative, got {scales}")

    def scale_at(self, step: int) -> float:
        """VOC scale for ``step`` — the value of the last threshold reached (``initial_scale`` before any)."""
        scale = self.initial_scale
        for threshold, value in self.schedule:
            if step < threshold:
                break
            scale = value
        return scale


@dataclass(frozen=True)
class CurriculumStage:
    """VOC scale, objective weights, and optional reset-distribution overrides."""

    voc_scale: float
    objective_weights: Mapping[str, float]
    reset_to_first_frame_probability: float | None = None
    immediate_first_frame_probability: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.voc_scale) or self.voc_scale < 0.0:
            raise ValueError(f"voc_scale must be finite and non-negative, got {self.voc_scale}")
        weights = dict(self.objective_weights)
        invalid_names = [name for name in weights if not isinstance(name, str) or not name]
        if invalid_names:
            raise ValueError(f"objective weight names must be non-empty strings, got {invalid_names}")
        invalid_weights = {name: value for name, value in weights.items() if not math.isfinite(value)}
        if invalid_weights:
            raise ValueError(f"objective weights must be finite, got {invalid_weights}")
        for name in ("reset_to_first_frame_probability", "immediate_first_frame_probability"):
            probability = getattr(self, name)
            if probability is not None and (not math.isfinite(probability) or not 0.0 <= probability <= 1.0):
                raise ValueError(f"{name} must be None or finite and in [0, 1], got {probability}")
        object.__setattr__(self, "objective_weights", weights)

    def weights_for(self, term_names: tuple[str, ...]) -> tuple[float, ...]:
        """Return weights in a strategy's fixed term order after exact name validation."""
        expected = set(term_names)
        actual = set(self.objective_weights)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(f"curriculum objective weights mismatch: missing={missing}, unexpected={unexpected}")
        return tuple(self.objective_weights[name] for name in term_names)


@dataclass(frozen=True)
class FixedCurriculum:
    """Stage schedule using the same threshold semantics as the reference run."""

    thresholds: tuple[int, ...]
    stages: tuple[CurriculumStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("stages must not be empty")
        if len(self.thresholds) != len(self.stages):
            raise ValueError(
                f"thresholds and stages must have equal length, got {len(self.thresholds)} and {len(self.stages)}"
            )
        if any(threshold < 0 for threshold in self.thresholds):
            raise ValueError(f"thresholds must be non-negative, got {self.thresholds}")
        if any(left >= right for left, right in zip(self.thresholds, self.thresholds[1:])):
            raise ValueError(f"thresholds must be strictly increasing, got {self.thresholds}")

    def stage_index(self, step: int) -> int:
        """Return the active stage; stage zero is active before the first threshold."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        return min(bisect_right(self.thresholds, step), len(self.stages) - 1)

    def stage_at(self, step: int) -> CurriculumStage:
        """Return the active immutable stage."""
        return self.stages[self.stage_index(step)]


def reference_curriculum() -> FixedCurriculum:
    """Return the fixed curriculum copied from W&B run ``sizirwre``."""
    thresholds = (2000, 3500, 5000, 6500, 8000, 9500, 11000, 12500, 14000, 15500)
    voc_scales = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0, 0.0)
    object_weights = (0.0, 0.1, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0, 20.0)
    stages = tuple(
        CurriculumStage(
            voc_scale=voc_scale,
            objective_weights={
                "object_keypoints": object_weight,
                "hand_keypoints": 0.25,
                "hand_joint_pos": 0.25,
                "contact_wrench_support": 10.0,
                "missed_contact": -1.0,
                "unintended_contact": -10.0,
                "termination": -100.0,
                "action_rate_l2": -0.005,
                "action_l2": -0.002,
                "contact_force_l2": 0.0,
                "force_closure": 0.0,
            },
        )
        for voc_scale, object_weight in zip(voc_scales, object_weights, strict=True)
    )
    return FixedCurriculum(thresholds=thresholds, stages=stages)
