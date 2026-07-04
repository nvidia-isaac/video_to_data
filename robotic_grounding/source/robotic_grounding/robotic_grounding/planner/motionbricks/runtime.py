# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX inference runtime for the MotionBricks whole-body planner.

``MotionBricksOnnxEngine`` drives three ONNX graphs per chunk -- root prediction
(``root``), masked-token pose prediction (``pose``), and decode (``decoder``)
with a host-side argmax in between -- and stitches overlapping
chunks into a full sequence with a triangular blend. Optionally pins the lower
body to a static pose and averages a half-stride second pass to soften seams.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from robotic_grounding.planner.motionbricks.canonicalization import (
    _derive_root_relative_ee_from_target,
    _heading_matrix_from_pose_frame,
    _packed_transforms_change_frame,
    _root_xy_local_to_shared,
    _root_xy_shared_to_local,
    _yaw_from_heading_matrix,
)


def _triangular_chunk_weights(num_frames: int) -> np.ndarray:
    """Symmetric overlap weights for feature-space chunk blending."""
    ramp = np.linspace(0.0, 1.0, num_frames // 2 + 1, dtype=np.float32)
    if num_frames % 2 == 0:
        weights = np.concatenate([ramp[:-1], ramp[::-1]])
    else:
        weights = np.concatenate([ramp, ramp[-2::-1]])
    return np.clip(weights[:num_frames], 0.1, 1.0)


# Lower-body body indices (hips/knees/ankles, both sides) in IsaacLab body order.
LOWER_BODY_BODY_INDICES_ISAACLAB: tuple[int, ...] = (
    1,
    2,
    4,
    5,
    7,
    8,
    10,
    11,
    14,
    15,
    18,
    19,
)


def _override_lower_body_in_place(
    transforms: np.ndarray, static_pose: np.ndarray
) -> None:
    """Pin lower-body body slots in ``transforms`` to ``static_pose``."""
    num_bodies = transforms.shape[-1] // 9
    F = transforms.shape[0]
    view = transforms.reshape(F, num_bodies, 9)
    static_view = static_pose[:F].reshape(F, num_bodies, 9)
    for idx in LOWER_BODY_BODY_INDICES_ISAACLAB:
        view[:, idx] = static_view[:, idx]


# Root EE key suffix that selects the absolute (non-root-relative) target frame.
ROOT_TARGET_EE_SUFFIX = "hand_ee_target_transforms_nonflat"


class MotionBricksOnnxEngine:
    """Numpy + onnxruntime MotionBricks chunked-autoregressive engine over 3 ONNX graphs."""

    def __init__(
        self, onnx_dir: str, providers: tuple[str, ...] = ("CPUExecutionProvider",)
    ) -> None:
        """Load the root/pose/decoder ONNX graphs, kinematics sidecar, and metadata."""
        onnx_path = Path(onnx_dir)
        self.meta = json.loads((onnx_path / "meta.json").read_text())
        arr = np.load(onnx_path / "arrays.npz", allow_pickle=True)
        m = self.meta
        self.nfpt = int(m["nfpt"])
        self.max_tokens = int(m["max_tokens"])
        self.max_frames = int(m["max_frames"])
        self.root_dim = int(m["root_dim"])
        self.pose_dim = int(m["pose_dim"])
        self.min_tokens = int(m["min_tokens"])
        self.num_heads = int(m["num_fsq_heads"])
        self.num_codes = int(m["num_codes_per_head"])
        self.pose_mask_id = int(m["pose_mask_id"])
        self.ee_dim = int(m["ee_dim"])
        self.root_ee_key = m["root_ee_key"]
        self.pose_ee_key = m["pose_ee_key"]
        self.derive_pose_ee_from_pred_root = bool(m["derive_pose_ee_from_pred_root"])
        # Absolute root-target EE key => run inference in a per-chunk local frame.
        self.root_target_ee = self.root_ee_key.endswith(ROOT_TARGET_EE_SUFFIX)

        # kinematics for features->qpos IK.
        self.kin = {
            "parents": arr["parents"].astype(np.int64),
            "dof_axis": arr["dof_axis"].astype(np.float64),
            "local_rotation_mat": arr["local_rotation_mat"].astype(np.float64),
            "num_bodies": int(arr["num_bodies"]),
            "num_dof": int(arr["num_dof"]),
        }
        self.body_reorder = arr["body_reorder"].astype(np.int64)

        so = ort.SessionOptions()
        prov = list(providers)
        self.sroot = ort.InferenceSession(
            str(onnx_path / "root.onnx"), so, providers=prov
        )
        self.spose = ort.InferenceSession(
            str(onnx_path / "pose.onnx"), so, providers=prov
        )
        self.sdec = ort.InferenceSession(
            str(onnx_path / "decoder.onnx"), so, providers=prov
        )

    def _run_inference(
        self,
        root_ee_input: np.ndarray,
        pose_ee_input: np.ndarray | None,
        num_tokens: int,
        start_root: np.ndarray,
        start_pose: np.ndarray,
    ) -> dict[str, Any]:
        """One forced-num_tokens chunk: root -> pose -> argmax -> decoder."""
        nfpt, root_dim, pose_dim = self.nfpt, self.root_dim, self.pose_dim
        max_tokens, max_frames = self.max_tokens, self.max_frames
        ee_dim = self.ee_dim
        pred_frames = num_tokens * nfpt

        # --- root inputs (sparse start keyframe window in first nfpt frames) ---
        rv = np.zeros((1, 2 * nfpt, root_dim), np.float32)
        hrv = np.zeros((1, 2 * nfpt), bool)
        pv = np.zeros((1, 2 * nfpt, pose_dim), np.float32)
        hpv = np.zeros((1, 2 * nfpt), bool)
        if start_root is not None:
            rv[0, :nfpt] = start_root
            hrv[0, :nfpt] = True
        if start_pose is not None:
            pv[0, :nfpt] = start_pose
            hpv[0, :nfpt] = True
        nt = np.array([num_tokens], np.int64)

        # dense root EE cond, padded to max_frames, masked to pred_frames.
        root_eev = np.zeros((1, max_frames, ee_dim), np.float32)
        root_eem = np.zeros((1, max_frames), bool)
        F_use = min(root_ee_input.shape[0], max_frames)
        root_eev[0, :F_use] = root_ee_input[:F_use]
        root_eem[0, :F_use] = True

        pred_root_full = self.sroot.run(
            ["pred_root_transforms"],
            dict(
                root_values=rv,
                has_root_values=hrv,
                pose_values=pv,
                has_pose_values=hpv,
                num_tokens_input=nt,
                ee_values=root_eev,
                ee_mask=root_eem,
            ),
        )[0][
            0
        ]  # [max_frames, root_dim]
        pred_root = pred_root_full[:pred_frames]

        # --- pose EE cond (derive root-relative from predicted root if configured) ---
        if self.derive_pose_ee_from_pred_root:
            if root_ee_input is None:
                pose_ee = None
            else:
                pose_ee = _derive_root_relative_ee_from_target(
                    root_ee_input[:pred_frames], pred_root
                )
        else:
            pose_ee = pose_ee_input

        # root_per_token: [1, max_tokens, nfpt*root_dim]
        root_for_pose = np.zeros((1, max_frames, root_dim), np.float32)
        root_for_pose[0, :pred_frames] = pred_root
        root_per_token = root_for_pose.reshape(1, max_tokens, nfpt * root_dim)

        pose_ee_padded = np.zeros((max_frames, ee_dim), np.float32)
        ee_token_mask = np.zeros((1, max_tokens), bool)
        if pose_ee is not None:
            F_pe = min(pose_ee.shape[0], max_frames)
            pose_ee_padded[:F_pe] = pose_ee[:F_pe]
            ee_mask_frame = np.zeros((1, max_frames), bool)
            ee_mask_frame[0, :F_pe] = True
            ee_token_mask = ee_mask_frame[:, ::nfpt][:, :max_tokens]
        ee_per_token = pose_ee_padded.reshape(1, max_tokens, nfpt * ee_dim)

        it = np.full((1, max_tokens, self.num_heads), self.pose_mask_id, np.int64)
        # start_pose seeds the pose keyframe conditioning for this chunk.
        pc = np.zeros((1, max_frames, pose_dim), np.float32)
        hpc = np.zeros((1, max_frames), bool)
        if start_pose is not None:
            pc[0, :nfpt] = start_pose
            hpc[0, :nfpt] = True
        ant = np.array([num_tokens], np.int64)

        pose_logits = self.spose.run(
            ["pose_logits"],
            dict(
                input_tokens=it,
                root_per_token=root_per_token.astype(np.float32),
                pose_cond=pc,
                has_pose_cond=hpc,
                actual_num_tokens=ant,
                ee_values=ee_per_token.astype(np.float32),
                ee_mask=ee_token_mask,
            ),
        )[
            0
        ]  # [1, max_tokens, heads, codes]

        pred_tokens = pose_logits.argmax(axis=-1).astype(np.int64)  # host argmax
        ee_frames = pose_ee_padded.reshape(1, 1, max_frames, ee_dim)
        pred_joints_full = self.sdec.run(
            ["pred_joints"], dict(token_indices=pred_tokens, ee_frames=ee_frames)
        )[
            0
        ]  # [1, max_frames, pose_dim]
        pred_joints = pred_joints_full[0, :pred_frames]

        return {
            "pred_root": pred_root,
            "pred_joints": pred_joints,
            "pred_num_tokens": int(num_tokens),
            "pred_frames": int(pred_frames),
        }

    def run_chunked_autoregressive_inference(
        self,
        root_ee_input: np.ndarray,
        seed_root: np.ndarray,
        seed_pose: np.ndarray,
        pose_ee_input: np.ndarray | None,
        chunk_tokens: int,
        overlap_tokens: int,
        static_pose_shared: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Overlapping forced-num_tokens AR windows with triangular blend.

        When ``static_pose_shared`` is given, pins lower-body bodies in the AR
        seed + prediction tail every chunk.
        """
        nfpt = self.nfpt
        max_tokens = self.max_tokens
        chunk_tokens = min(max(int(chunk_tokens), 2), max_tokens)
        overlap_tokens = max(1, min(int(overlap_tokens), chunk_tokens - 1))
        overlap_frames = overlap_tokens * nfpt
        chunk_frames = chunk_tokens * nfpt
        stride_frames = chunk_frames - overlap_frames

        total_frames = (root_ee_input.shape[0] // nfpt) * nfpt
        if total_frames < nfpt:
            raise ValueError(
                f"Need at least {nfpt} frames, got {root_ee_input.shape[0]}"
            )

        root_dim, pose_dim = self.root_dim, self.pose_dim
        root_accum = np.zeros((total_frames, root_dim), np.float32)
        pose_accum = np.zeros((total_frames, pose_dim), np.float32)
        weight_accum = np.zeros((total_frames, 1), np.float32)

        previous_result = None
        previous_range = None
        chunk_infos: list[dict] = []
        use_chunk_local_frame = self.root_target_ee

        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            end = start + ((end - start) // nfpt) * nfpt
            if end - start < nfpt:
                break

            window_num_tokens = (end - start) // nfpt
            root_ee_window = root_ee_input[start:end]
            pose_ee_window = (
                pose_ee_input[start:end] if pose_ee_input is not None else None
            )

            if previous_result is None:
                start_root_shared = seed_root[:nfpt].copy()
                start_pose_shared = seed_pose[:nfpt].copy()
            else:
                assert previous_range is not None
                prev_start, _ = previous_range
                local_start = start - prev_start
                local_end = local_start + nfpt
                if local_start < 0 or local_end > previous_result["pred_frames"]:
                    raise RuntimeError(
                        f"Chunk {start}:{end} not covered by previous range {previous_range}"
                    )
                start_root_shared = previous_result["pred_root"][
                    local_start:local_end
                ].copy()
                start_pose_shared = previous_result["pred_joints"][
                    local_start:local_end
                ].copy()

            if static_pose_shared is not None:
                _override_lower_body_in_place(
                    start_pose_shared, static_pose_shared[start : start + nfpt]
                )

            if use_chunk_local_frame:
                chunk_origin_xy = np.asarray(start_root_shared[0, :2], dtype=np.float32)
                chunk_heading_mat = _heading_matrix_from_pose_frame(
                    start_pose_shared[0]
                ).astype(np.float32)
                start_root = _root_xy_shared_to_local(
                    start_root_shared, chunk_origin_xy, chunk_heading_mat
                )
                start_pose = _packed_transforms_change_frame(
                    start_pose_shared,
                    chunk_heading_mat,
                    inverse_heading=True,
                    positions_are_root_relative=True,
                )
                root_ee_window_model = _packed_transforms_change_frame(
                    root_ee_window,
                    chunk_heading_mat,
                    origin_xy=chunk_origin_xy if self.root_target_ee else None,
                    inverse_heading=True,
                    positions_are_root_relative=not self.root_target_ee,
                )
                pose_ee_window_model = (
                    _packed_transforms_change_frame(
                        pose_ee_window,
                        chunk_heading_mat,
                        origin_xy=None,
                        inverse_heading=True,
                        positions_are_root_relative=True,
                    )
                    if pose_ee_window is not None
                    else None
                )
            else:
                chunk_origin_xy = np.zeros(2, np.float32)
                chunk_heading_mat = np.eye(3, np.float32)
                start_root = start_root_shared
                start_pose = start_pose_shared
                root_ee_window_model = root_ee_window
                pose_ee_window_model = pose_ee_window
            chunk_heading_yaw = _yaw_from_heading_matrix(chunk_heading_mat)

            result = self._run_inference(
                root_ee_input=root_ee_window_model,
                pose_ee_input=pose_ee_window_model,
                num_tokens=window_num_tokens,
                start_root=start_root,
                start_pose=start_pose,
            )

            local_n = min(result["pred_frames"], end - start)
            output_num_tokens = int(np.ceil(local_n / nfpt))
            pred_root_shared = (
                _root_xy_local_to_shared(
                    result["pred_root"][:local_n], chunk_origin_xy, chunk_heading_mat
                )
                if use_chunk_local_frame
                else result["pred_root"][:local_n]
            )
            pred_joints_shared = (
                _packed_transforms_change_frame(
                    result["pred_joints"][:local_n],
                    chunk_heading_mat,
                    inverse_heading=False,
                    positions_are_root_relative=True,
                )
                if use_chunk_local_frame
                else result["pred_joints"][:local_n]
            )

            if static_pose_shared is not None:
                _override_lower_body_in_place(
                    pred_joints_shared, static_pose_shared[start : start + local_n]
                )

            weights = _triangular_chunk_weights(local_n)[:, None]
            root_accum[start : start + local_n] += pred_root_shared * weights
            pose_accum[start : start + local_n] += pred_joints_shared * weights
            weight_accum[start : start + local_n] += weights

            chunk_infos.append(
                {
                    "start_frame": start,
                    "end_frame": start + local_n,
                    "num_tokens": output_num_tokens,
                    "window_num_tokens": int(window_num_tokens),
                    "predicted_num_tokens": int(result["pred_num_tokens"]),
                    "chunk_num_tokens_mode": "forced",
                    "chunk_origin_xy": chunk_origin_xy.tolist(),
                    "chunk_heading_yaw_rad": chunk_heading_yaw,
                    "chunk_local_frame_enabled": bool(use_chunk_local_frame),
                }
            )

            previous_result = {
                "pred_root": pred_root_shared,
                "pred_joints": pred_joints_shared,
                "pred_frames": local_n,
            }
            previous_range = (start, start + local_n)
            if start + local_n >= total_frames:
                break
            start += stride_frames

        valid = weight_accum[:, 0] > 0
        if not np.all(valid):
            last_valid = int(np.nonzero(valid)[0][-1]) + 1
            root_accum = root_accum[:last_valid]
            pose_accum = pose_accum[:last_valid]
            weight_accum = weight_accum[:last_valid]

        pred_root = root_accum / np.clip(weight_accum, 1e-8, None)
        pred_joints = pose_accum / np.clip(weight_accum, 1e-8, None)
        if static_pose_shared is not None:
            _override_lower_body_in_place(
                pred_joints, static_pose_shared[: pred_joints.shape[0]]
            )

        return {
            "pred_root": pred_root,
            "pred_joints": pred_joints,
            "pred_num_tokens": int(np.ceil(pred_root.shape[0] / nfpt)),
            "pred_frames": int(pred_root.shape[0]),
            "chunk_infos": chunk_infos,
        }


def blend_two_passes(
    pred_a: dict[str, Any],
    pred_b: dict[str, Any],
    *,
    offset_b: int,
    total_frames: int,
) -> dict[str, Any]:
    """Average two AR passes sharing the same shared-frame indexing."""
    F = min(total_frames, max(pred_a["pred_frames"], offset_b + pred_b["pred_frames"]))
    accum_root = np.zeros((F, pred_a["pred_root"].shape[-1]), np.float32)
    accum_pose = np.zeros((F, pred_a["pred_joints"].shape[-1]), np.float32)
    count = np.zeros((F, 1), np.float32)
    n_a = min(pred_a["pred_frames"], F)
    accum_root[:n_a] += pred_a["pred_root"][:n_a]
    accum_pose[:n_a] += pred_a["pred_joints"][:n_a]
    count[:n_a] += 1.0
    n_b = min(pred_b["pred_frames"], F - offset_b)
    if n_b > 0:
        accum_root[offset_b : offset_b + n_b] += pred_b["pred_root"][:n_b]
        accum_pose[offset_b : offset_b + n_b] += pred_b["pred_joints"][:n_b]
        count[offset_b : offset_b + n_b] += 1.0
    count = np.clip(count, 1e-8, None)
    return {
        "pred_root": accum_root / count,
        "pred_joints": accum_pose / count,
        "pred_num_tokens": pred_a.get("pred_num_tokens", 0),
        "pred_frames": int(F),
        "chunk_infos": (pred_a.get("chunk_infos") or [])
        + (pred_b.get("chunk_infos") or []),
    }
