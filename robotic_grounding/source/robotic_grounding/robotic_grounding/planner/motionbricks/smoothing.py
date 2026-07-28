# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qpos smoothing: global Hamming smoothing and targeted chunk-seam smoothing.

Positions and joints are smoothed with a Hamming window; root quaternions are
sign-aligned before smoothing and re-normalized. Chunk seams (where overlapping
autoregressive windows meet) get extra Gaussian-masked smoothing.
"""
from __future__ import annotations

import numpy as np


def _hamming_smooth(signal: np.ndarray, half_width: int) -> np.ndarray:
    if half_width <= 0 or signal.shape[0] <= 2:
        return signal.copy()
    signal = np.asarray(signal, dtype=np.float32)
    padded = np.concatenate(
        [
            np.repeat(signal[:1], half_width, 0),
            signal,
            np.repeat(signal[-1:], half_width, 0),
        ],
        0,
    )
    kernel = np.hamming(2 * half_width + 1).astype(np.float32)
    kernel /= kernel.sum()
    flat = padded.reshape(padded.shape[0], -1)
    out = np.empty((signal.shape[0], flat.shape[1]), dtype=np.float32)
    for i in range(signal.shape[0]):
        out[i] = (flat[i : i + kernel.shape[0]] * kernel[:, None]).sum(0)
    return out.reshape(signal.shape)


def _smooth_quat_wxyz(quat: np.ndarray, half_width: int) -> np.ndarray:
    if half_width <= 0 or quat.shape[0] <= 2:
        return quat.copy()
    aligned = np.asarray(quat, dtype=np.float32).copy()
    for i in range(1, aligned.shape[0]):
        if float(np.dot(aligned[i - 1], aligned[i])) < 0.0:
            aligned[i] *= -1.0
    sm = _hamming_smooth(aligned, half_width)
    return (
        sm / np.clip(np.linalg.norm(sm, axis=-1, keepdims=True), 1e-8, None)
    ).astype(np.float32)


def _nlerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32).copy()
    q1 = np.where(np.sum(q0 * q1, -1, keepdims=True) < 0.0, -q1, q1)
    out = q0 * (1.0 - alpha) + q1 * alpha
    return (
        out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)
    ).astype(np.float32)


def _smooth_g1_qpos_wxyz(
    qpos: np.ndarray,
    pos_width: int = 3,
    quat_width: int = 3,
    joint_width: int = 2,
) -> np.ndarray:
    out = np.asarray(qpos, dtype=np.float32).copy()
    out[:, :3] = _hamming_smooth(out[:, :3], pos_width)
    out[:, 3:7] = _smooth_quat_wxyz(out[:, 3:7], quat_width)
    out[:, 7:] = _hamming_smooth(out[:, 7:], joint_width)
    return out


def _chunk_boundary_centers(chunk_infos: list[dict] | None) -> list[int]:
    """Frame index at the center of each pair of overlapping chunks."""
    if not chunk_infos:
        return []
    centers: list[int] = []
    for prev, cur in zip(chunk_infos[:-1], chunk_infos[1:], strict=True):
        overlap_start = int(cur["start_frame"])
        overlap_end = min(int(prev["end_frame"]), int(cur["end_frame"]))
        if overlap_end > overlap_start:
            centers.append(overlap_start + (overlap_end - overlap_start) // 2)
        else:
            centers.append(overlap_start)
    return centers


def smooth_qpos_global(qpos: np.ndarray, *, nfpt: int) -> np.ndarray:
    """Globally smooth qpos with Hamming windows scaled to the token size."""
    return _smooth_g1_qpos_wxyz(
        qpos, pos_width=2, quat_width=2, joint_width=max(1, nfpt // 2)
    )


def smooth_qpos_at_boundaries(
    qpos: np.ndarray, boundary_centers: list[int], *, nfpt: int
) -> np.ndarray:
    """Apply heavier smoothing near chunk seams, Gaussian-masked around each center."""
    if not boundary_centers:
        return qpos.copy()
    raw = np.asarray(qpos, dtype=np.float32)
    boundary_radius = max(3 * nfpt, 6)
    smooth = _smooth_g1_qpos_wxyz(
        raw,
        pos_width=max(boundary_radius // 2, 2),
        quat_width=max(boundary_radius // 2, 2),
        joint_width=boundary_radius,
    )
    T = raw.shape[0]
    frames = np.arange(T, dtype=np.float32)
    mask = np.zeros((T, 1), dtype=np.float32)
    sigma = max(float(nfpt) * 1.5, 1.0)
    for center in boundary_centers:
        gaussian = np.exp(-0.5 * ((frames - float(center)) / sigma) ** 2)
        mask = np.maximum(mask, gaussian.reshape(T, 1).astype(np.float32))
    out = raw.copy()
    out[:, :3] = raw[:, :3] * (1.0 - mask) + smooth[:, :3] * mask
    out[:, 7:] = raw[:, 7:] * (1.0 - mask) + smooth[:, 7:] * mask
    out[:, 3:7] = _nlerp_quat_wxyz(raw[:, 3:7], smooth[:, 3:7], mask)
    return out.astype(np.float32)
