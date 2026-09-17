"""FoundationPose scorer/refiner backend selection."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v2d.foundation_pose.lib.trt_engine import ensure_engines, runtime_manifest


NVIDIA_TENSORRT = "nvidia_tensorrt"
NVLABS_PYTORCH = "nvlabs_pytorch"
DEFAULT_BACKEND = NVIDIA_TENSORRT
SUPPORTED_BACKENDS = (NVIDIA_TENSORRT, NVLABS_PYTORCH)


@dataclass
class PredictorBackend:
    scorer: Any
    refiner: Any
    runtime_info: dict


def _nvlabs_weights_dir(weights_dir: str | os.PathLike[str]) -> Path:
    root = Path(weights_dir)
    nested = root / NVLABS_PYTORCH
    if nested.is_dir():
        return nested
    # Retain compatibility with already-provisioned weight directories while
    # new downloads use the named backend subdirectory.
    return root


def create_predictor_backend(
    backend: str = DEFAULT_BACKEND,
    weights_dir: str | os.PathLike[str] = "",
) -> PredictorBackend:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported FoundationPose backend {backend!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )

    if backend == NVIDIA_TENSORRT:
        engines = ensure_engines(weights_dir)
        from v2d.foundation_pose.lib.trt_predictors import (
            TensorRTPoseRefinePredictor,
            TensorRTScorePredictor,
        )

        return PredictorBackend(
            scorer=TensorRTScorePredictor(engines["score"]),
            refiner=TensorRTPoseRefinePredictor(engines["refine"]),
            runtime_info=runtime_manifest(weights_dir, engines),
        )

    fp_dir = Path(__file__).resolve().parent / "FoundationPose"
    if str(fp_dir) not in sys.path:
        sys.path.insert(0, str(fp_dir))
    resolved_weights = _nvlabs_weights_dir(weights_dir)
    os.environ["FOUNDATIONPOSE_WEIGHTS_DIR"] = str(resolved_weights)
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    return PredictorBackend(
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        runtime_info={
            "backend": NVLABS_PYTORCH,
            "weights_dir": str(resolved_weights),
            "scorer_run": "2024-01-11-20-02-45",
            "refiner_run": "2023-10-28-18-33-37",
        },
    )
