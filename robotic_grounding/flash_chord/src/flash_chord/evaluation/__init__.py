# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object-tracking evaluation: deterministic cohort recording and CPU metrics."""

__all__ = [
    "EvaluationOutcome",
    "ObjectTrackingMetrics",
    "TrackingThresholds",
    "evaluate_checkpoint",
    "evaluation_scalars",
    "write_evaluation_report",
]


def __getattr__(name: str) -> object:
    """Resolve lazily so importing metrics does not pull in Warp or JAX."""
    if name in {"ObjectTrackingMetrics", "TrackingThresholds"}:
        from flash_chord.evaluation import metrics

        return getattr(metrics, name)
    if name in set(__all__):
        from flash_chord.evaluation import harness

        return getattr(harness, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
