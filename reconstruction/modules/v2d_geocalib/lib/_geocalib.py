# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GeoCalib model loader and serialization helpers."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from v2d.common.datatypes import CameraDistortion, CameraIntrinsics


_MODEL_CACHE: dict[tuple[str, str | None, str], object] = {}


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _to_builtin(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


def _get_model(weights_path: str | None, weights: str, device: str):
    cache_key = (weights, os.path.abspath(weights_path) if weights_path else None, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if weights_path:
        weights_path = os.path.abspath(weights_path)
        if os.path.isdir(weights_path):
            os.environ["TORCH_HOME"] = weights_path
            model_weights = weights
        elif os.path.isfile(weights_path):
            model_weights = weights_path
        else:
            raise FileNotFoundError(f"GeoCalib weights path not found: {weights_path}")
    else:
        model_weights = weights

    from geocalib import GeoCalib

    print(f"Loading GeoCalib weights={model_weights!r} on {device}")
    model = GeoCalib(weights=model_weights).to(device).eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _camera_to_intrinsics(camera: Any) -> CameraIntrinsics:
    size = _array(camera.size).reshape(-1, 2)[0]
    focal = _array(camera.f).reshape(-1, 2)[0]
    center = _array(camera.c).reshape(-1, 2)[0]
    return CameraIntrinsics(
        fx=float(focal[0]),
        fy=float(focal[1]),
        cx=float(center[0]),
        cy=float(center[1]),
        width=int(round(float(size[0]))),
        height=int(round(float(size[1]))),
    )


def _camera_to_distortion(camera: Any, camera_model: str) -> CameraDistortion:
    if camera_model == "pinhole" or not hasattr(camera, "dist"):
        return CameraDistortion(model="pinhole", params=[])
    dist = _array(camera.dist).reshape(-1)
    if hasattr(camera, "num_dist_params"):
        dist = dist[: int(camera.num_dist_params())]
    model = {
        "simple_radial": "geocalib_simple_radial",
        "radial": "geocalib_radial",
        "simple_divisional": "geocalib_simple_divisional",
    }.get(camera_model, f"geocalib_{camera_model}")
    return CameraDistortion(model=model, params=[float(v) for v in dist])


def _gravity_record(gravity: Any) -> dict:
    vec = _array(gravity.vec3d).reshape(-1, 3)[0]
    roll = _scalar(gravity.roll)
    pitch = _scalar(gravity.pitch)
    R = _array(gravity.R).reshape(-1, 3, 3)[0]
    return {
        "vector_camera": [float(v) for v in vec],
        "roll_rad": roll,
        "pitch_rad": pitch,
        "roll_deg": float(np.degrees(roll)),
        "pitch_deg": float(np.degrees(pitch)),
        "rotation_matrix": R.tolist(),
    }


def predict_calibration(
    image: torch.Tensor,
    weights_path: str | None,
    weights: str = "pinhole",
    camera_model: str = "pinhole",
    device: str | None = None,
) -> tuple[CameraIntrinsics, CameraDistortion, dict]:
    """Run GeoCalib on one image tensor in RGB CHW/BCHW [0, 1] format."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_model(weights_path, weights=weights, device=device)
    image = image.to(device)
    result = model.calibrate(image, camera_model=camera_model)
    intrinsics = _camera_to_intrinsics(result["camera"])
    distortion = _camera_to_distortion(result["camera"], camera_model)
    gravity = _gravity_record(result["gravity"])
    extra = {
        "weights": weights,
        "camera_model": camera_model,
        "gravity": gravity,
        "uncertainty": {
            k: _to_builtin(v)
            for k, v in result.items()
            if "uncertainty" in k
        },
        "covariance": _to_builtin(result.get("covariance")),
    }
    return intrinsics, distortion, extra


def aggregate_calibrations(records: list[dict]) -> tuple[CameraIntrinsics, CameraDistortion, dict]:
    """Median intrinsics/distortion and normalized mean gravity over samples."""
    if not records:
        raise ValueError("No GeoCalib records to aggregate")

    intrinsics = [CameraIntrinsics.from_dict(r["intrinsics"]) for r in records]
    distortions = [CameraDistortion.from_dict(r["distortion"]) for r in records]
    dist_models = {d.model for d in distortions}
    if len(dist_models) != 1:
        raise ValueError(f"Cannot aggregate mixed distortion models: {dist_models}")

    dist_len = len(distortions[0].params)
    if any(len(d.params) != dist_len for d in distortions):
        raise ValueError("Cannot aggregate GeoCalib distortions with different lengths")

    gravity_vectors = np.asarray(
        [r["gravity"]["vector_camera"] for r in records],
        dtype=np.float64,
    )
    gravity = gravity_vectors.mean(axis=0)
    norm = float(np.linalg.norm(gravity))
    if norm < 1e-8:
        raise RuntimeError("GeoCalib gravity estimates cancelled out during aggregation")
    gravity /= norm

    fx = float(np.median([k.fx for k in intrinsics]))
    fy = float(np.median([k.fy for k in intrinsics]))
    cx = float(np.median([k.cx for k in intrinsics]))
    cy = float(np.median([k.cy for k in intrinsics]))
    width = int(round(float(np.median([k.width for k in intrinsics]))))
    height = int(round(float(np.median([k.height for k in intrinsics]))))
    if dist_len:
        params = np.median(
            np.asarray([d.params for d in distortions], dtype=np.float64),
            axis=0,
        ).tolist()
    else:
        params = []

    agg_intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
    )
    agg_distortion = CameraDistortion(
        model=next(iter(dist_models)),
        params=[float(v) for v in params],
    )
    aggregate = {
        "gravity": {
            "vector_camera": [float(v) for v in gravity],
            "sample_vectors_camera": gravity_vectors.tolist(),
        },
        "num_samples": len(records),
    }
    return agg_intrinsics, agg_distortion, aggregate

