"""AnyCalib model loader + thin inference helper.

AnyCalib (Tirado-Garín & Civera, 2024) is a learned single-view camera
calibrator that outputs intrinsics + distortion under a chosen camera model.
We expose a single ``predict_calibration`` entry point that maps AnyCalib's
output into our typed ``CameraIntrinsics`` + ``CameraDistortion`` pair.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from v2d.common.datatypes import CameraDistortion, CameraIntrinsics


# AnyCalib cam_id → (CameraDistortion.model, distortion-coeff count).
# The intrinsic head emits [fx, fy, cx, cy, *dist_params]; the distortion model
# determines how the trailing params map onto OpenCV conventions.
_CAM_ID_TO_DISTORTION: dict[str, tuple[str, int]] = {
    "pinhole":    ("pinhole",        0),
    "radial:1":   ("opencv",         1),  # k1
    "radial:2":   ("opencv",         2),  # k1, k2
    "radial:4":   ("opencv",         4),  # k1, k2, k3, k4
    "kb:4":       ("opencv_fisheye", 4),  # k1, k2, k3, k4 (Kannala-Brandt)
}

_model_cache: dict[str, object] = {}


def _get_model(weights_path: str, model_id: str = "anycalib_gen"):
    """Load (and cache) an AnyCalib model.

    ``weights_path`` is treated as the ``TORCH_HOME`` cache directory (AnyCalib
    pulls both its checkpoint and the DINOv2 backbone via ``torch.hub``).
    ``model_id`` selects the variant (anycalib_pinhole / gen / dist / edit).
    """
    cache_key = f"{model_id}::{weights_path}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    weights_path = os.path.abspath(weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"AnyCalib weights cache not found at {weights_path}")
    os.environ["TORCH_HOME"] = weights_path

    from anycalib import AnyCalib  # imported after TORCH_HOME is set
    print(f"Loading AnyCalib '{model_id}' (TORCH_HOME={weights_path})")
    model = AnyCalib(model_id=model_id).to("cuda").eval()
    _model_cache[cache_key] = model
    return model


def _to_input_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 RGB image to NCHW float32 tensor in [0, 1] on CUDA."""
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to("cuda")


def cam_id_distortion_spec(cam_id: str) -> tuple[str, int]:
    """Return (CameraDistortion.model, num_dist_params) for a given AnyCalib cam_id."""
    if cam_id not in _CAM_ID_TO_DISTORTION:
        raise ValueError(
            f"Unsupported cam_id '{cam_id}'. Supported: {sorted(_CAM_ID_TO_DISTORTION)}"
        )
    return _CAM_ID_TO_DISTORTION[cam_id]


def predict_calibration(
    image: np.ndarray,
    weights_path: str,
    cam_id: str = "kb:4",
    model_id: str = "anycalib_gen",
) -> tuple[CameraIntrinsics, CameraDistortion]:
    """Estimate intrinsics + distortion from a single RGB image.

    Args:
        image:        HWC uint8 RGB array.
        weights_path: Path to AnyCalib checkpoint (file or HF snapshot dir).
        cam_id:       AnyCalib camera-model id; default ``"kb:4"`` (4-param
                      Kannala-Brandt fisheye). Other supported ids in
                      ``_CAM_ID_TO_DISTORTION``.
        model_id:     Which AnyCalib weights variant the checkpoint corresponds
                      to. ``anycalib_gen`` accepts any cam_id; the more
                      specialised variants are constrained.

    Returns:
        ``(CameraIntrinsics, CameraDistortion)`` for the camera that produced
        the image. Both share the same image (width, height).
    """
    model = _get_model(weights_path, model_id=model_id)
    h, w = image.shape[:2]
    input_tensor = _to_input_tensor(image)

    with torch.no_grad():
        out = model.predict(input_tensor, cam_id=cam_id)

    params = out["intrinsics"][0].detach().cpu().numpy().astype(np.float64)
    if params.shape[0] < 4:
        raise RuntimeError(f"AnyCalib returned {params.shape[0]} params, expected ≥ 4")

    fx, fy, cx, cy = (float(v) for v in params[:4])
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h))

    dist_model, n_dist = cam_id_distortion_spec(cam_id)
    dist_params = [float(v) for v in params[4:4 + n_dist]] if n_dist > 0 else []
    distortion = CameraDistortion(model=dist_model, params=dist_params)
    return intrinsics, distortion


def aggregate_calibrations(
    calibrations: list[tuple[CameraIntrinsics, CameraDistortion]],
) -> tuple[CameraIntrinsics, CameraDistortion]:
    """Combine multiple per-frame calibrations into one robust estimate (median)."""
    if not calibrations:
        raise ValueError("No calibrations to aggregate")

    intrinsics_list = [c[0] for c in calibrations]
    distortions = [c[1] for c in calibrations]

    models = {d.model for d in distortions}
    if len(models) != 1:
        raise ValueError(f"Cannot aggregate mixed distortion models: {models}")
    dist_model = next(iter(models))

    width = intrinsics_list[0].width
    height = intrinsics_list[0].height

    fx = float(np.median([k.fx for k in intrinsics_list]))
    fy = float(np.median([k.fy for k in intrinsics_list]))
    cx = float(np.median([k.cx for k in intrinsics_list]))
    cy = float(np.median([k.cy for k in intrinsics_list]))

    if distortions[0].params:
        stacked = np.stack([np.asarray(d.params, dtype=np.float64) for d in distortions])
        dist_params = np.median(stacked, axis=0).tolist()
    else:
        dist_params = []

    return (
        CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height),
        CameraDistortion(model=dist_model, params=dist_params),
    )
