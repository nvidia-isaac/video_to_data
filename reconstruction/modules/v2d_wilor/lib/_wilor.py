# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Singleton WiLoR loader + inference helpers.

Wraps the upstream pipeline:
    wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline.WiLorHandPose3dEstimationPipeline

WiLoR is end-to-end: given an image it returns both hand bboxes (via its
internal YOLO detector) and per-hand MANO params. We expose two entry points:

    run_wilor_detect(image, ...)            → detector + reconstructor on whole image
    run_wilor_on_bboxes(image, bboxes, ...) → reconstructor only, caller-supplied bboxes

Both return a list of dicts in our unified per-detection schema (see
``image_to_hands.py`` for the JSON layout).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from v2d.common.datatypes import BoundingBox

_PIPE = None


def get_pipeline(weights_dir: str, dtype: torch.dtype = torch.float16):
    """Return a singleton WiLoR pipeline writing/reading weights under ``weights_dir``.

    The upstream pipeline auto-downloads its checkpoints on first instantiation.
    By default it writes them next to the installed package, which is not
    writable for the non-root container user. The ``wilor_pretrained_dir``
    kwarg redirects all four files (mano_mean_params.npz, MANO_RIGHT.pkl,
    wilor_final.ckpt, detector.pt) under ``<weights_dir>/pretrained_models/``.
    """
    global _PIPE
    if _PIPE is None:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _PIPE = WiLorHandPose3dEstimationPipeline(
            device=device, dtype=dtype, wilor_pretrained_dir=weights_dir,
        )
    return _PIPE


def _record_from_pred(pred: dict, image_size: tuple[int, int]) -> dict:
    """Normalize a WiLoR pipeline prediction into our unified record."""
    wp = pred["wilor_preds"]
    bbox = pred["hand_bbox"]
    is_right = bool(int(pred["is_right"]))
    W, H = image_size

    # All wilor_preds entries are numpy arrays with leading batch dim of 1.
    betas         = np.asarray(wp["betas"]).reshape(-1)           # (10,)
    global_orient = np.asarray(wp["global_orient"]).reshape(-1)   # (3,)
    hand_pose     = np.asarray(wp["hand_pose"]).reshape(-1)       # (45,) = 15*3
    cam_t_full    = np.asarray(wp["pred_cam_t_full"]).reshape(-1) # (3,)

    # WiLoR's pipeline post-processes left-hand axis-angle by negating the y
    # and z components — this expresses the rotation in a YZ-plane-mirrored
    # frame. Undo that so the stored params live in the model's right-hand
    # canonical frame (HaMeR convention). The renderer / downstream consumers
    # use a right-MANO model and mirror vertex x for left hands.
    if not is_right:
        global_orient = global_orient * np.array([1.0, -1.0, -1.0])
        hand_pose     = hand_pose.reshape(-1, 3) * np.array([1.0, -1.0, -1.0])
        hand_pose     = hand_pose.reshape(-1)

    return {
        "is_right": is_right,
        "score":    1.0,
        "bbox": BoundingBox(
            x0=float(bbox[0]), y0=float(bbox[1]),
            x1=float(bbox[2]), y1=float(bbox[3]),
        ).to_dict(),
        "mano": {
            "betas":         betas.tolist(),
            "global_orient": global_orient.tolist(),
            "hand_pose":     hand_pose.tolist(),
        },
        "camera": {
            "pred_cam_t_full":     cam_t_full.tolist(),
            "scaled_focal_length": float(wp["scaled_focal_length"]),
        },
        "image_size": [int(W), int(H)],
    }


def run_wilor_detect(
    image: np.ndarray, weights_dir: str, dtype: torch.dtype = torch.float16,
) -> List[dict]:
    """Detect + reconstruct hands on a whole image (RGB uint8, HxWx3)."""
    pipe = get_pipeline(weights_dir, dtype=dtype)
    H, W = image.shape[:2]
    preds = pipe.predict(image)
    return [_record_from_pred(p, (W, H)) for p in preds]


def run_wilor_on_bboxes(
    image: np.ndarray,
    bboxes: Sequence[BoundingBox],
    is_right_flags: Sequence[Optional[bool]],
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> List[dict]:
    """Reconstruct hands given caller-supplied bboxes.

    ``is_right_flags[i]`` must be a bool — wilor-mini's ``predict_with_bboxes``
    does not accept an "unknown" sentinel and uses the value to drive the
    left/right-hand mirroring. If you don't know handedness, run detector mode.
    """
    pipe = get_pipeline(weights_dir, dtype=dtype)
    H, W = image.shape[:2]
    if any(r is None for r in is_right_flags):
        raise ValueError(
            "run_wilor_on_bboxes requires known handedness for each bbox "
            "(wilor-mini does not predict it from a pre-cropped bbox)."
        )
    boxes_xyxy = np.array(
        [[b.x0, b.y0, b.x1, b.y1] for b in bboxes], dtype=np.float32,
    )
    is_rights = np.array([int(bool(r)) for r in is_right_flags], dtype=np.int32)
    preds = pipe.predict_with_bboxes(image, bboxes=boxes_xyxy, is_rights=is_rights)
    return [_record_from_pred(p, (W, H)) for p in preds]
