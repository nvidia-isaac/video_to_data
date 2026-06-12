# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-body motion inference from EE positions.

Loads the core inferencer from a torch.package archive (planner_agent.pkg)
for the predict() neural network call, and implements all pre/post processing
locally. The bundled package contains the model weights, motion representation
modules, and qpos converter — no external dependencies needed.
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from robotic_grounding.planner.mfm.chunk_runner import (
    run_chunked_inference,
    stitch_predictions,
)
from robotic_grounding.planner.mfm.data_adapters import (
    arrays_to_mfm_features,
    build_gt_qpos,
    resample_qpos,
)
from robotic_grounding.planner.mfm.motion_reps import quaternion_to_cont6d_np

_ASSETS_DIR = Path(__file__).parent / "assets"

# Coordinate transform: MuJoCo z-up → model y-up
_R_ZUP_TO_YUP = Rotation.from_euler("x", -np.pi / 2)
_R_X_TO_Z_FWD = Rotation.from_euler("z", -np.pi / 2)
_MJ_TO_MODEL = _R_ZUP_TO_YUP * _R_X_TO_Z_FWD

# Palm correction inverses (from retargeting)
_R_PALM_LEFT_INV = Rotation.from_matrix(
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
).inv()
_R_PALM_RIGHT_INV = Rotation.from_matrix(
    np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
).inv()


class MotionInferenceAgent:
    """Whole-body motion inference: EE targets → full-body qpos.

    Usage:
        agent = MotionInferenceAgent(device="cuda")
        result = agent.infer_from_ee_positions(
            root_pos, root_wxyz, left_ee_pos, left_ee_quat,
            right_ee_pos, right_ee_quat,
        )
        qpos = result["autoregressive"]["qpos"]  # (T, 36)
    """

    def __init__(self, device: str = "cuda"):
        pkg_path = _ASSETS_DIR / "models" / "planner_agent.pkg"
        if not pkg_path.exists():
            raise FileNotFoundError(
                f"Package not found at {pkg_path}. Run the export script first."
            )

        from torch.package import PackageImporter  # noqa: PLC0415

        imp = PackageImporter(str(pkg_path))
        bundle = imp.load_pickle("bundle", "bundle.pkl")

        self._inferencer = bundle["inferencer"]
        self._inferencer.to(device)
        self._inferencer._device = device  # .to() doesn't update this string attr
        self._inferencer.eval()
        self._device = device

        self._converter = bundle["converter"]
        self._converter.to(device)
        self._converter.eval()

        self._skeleton = bundle["skeleton"]
        # Override skeleton folder to our local assets so T-pose files resolve
        # without depending on CWD or external paths
        self._skeleton.folder = str(_ASSETS_DIR / "skeleton")
        self._global_motion_rep = bundle["global_motion_rep"]
        self._local_motion_rep = bundle["local_motion_rep"]
        self._motion_rep = bundle["motion_rep"]

        self._nfpt = bundle["nfpt"]
        self._model_fps = bundle["model_fps"]
        self._max_tokens = bundle["max_tokens"]
        self._min_tokens = bundle["min_tokens"]
        self._start_root_only = bundle["start_root_only"]
        self._xml_path = str(_ASSETS_DIR.parent / "assets" / "mujoco" / "g1_29dof.xml")

        # Build the models dict expected by run_one_chunk
        self._models = {
            "pose": self._inferencer._mmm_pose_model,
            "root": self._inferencer._mmm_root_model,
        }

        # Load helper functions from the bundled package namespace
        self._get_ee_pose_indices_fn = imp.import_module(
            "mfm.core.utils.mask_cond"
        ).get_ee_pose_indices

    @property
    def nfpt(self) -> int:
        return self._nfpt

    @property
    def model_fps(self) -> int:
        return self._model_fps

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def infer_from_ee_positions(
        self,
        root_pos: np.ndarray,
        root_wxyz: np.ndarray,
        left_ee_pos: np.ndarray,
        left_ee_quat_wxyz: np.ndarray,
        right_ee_pos: np.ndarray,
        right_ee_quat_wxyz: np.ndarray,
        root_height_override: float | None = None,
        z_offset: float = 0.0,
        src_fps: float | None = None,
        max_chunk_tokens: int = 6,
        modes: tuple[str, ...] = ("autoregressive",),
        smooth: bool = True,
        half_stride_blend: bool = True,
    ) -> dict:
        """Run whole-body inference from end-effector positions.

        All inputs are in MuJoCo z-up world frame.
        Output qpos layout: [pos(3), xyzw_quat(4), joints(29)] = 36 dims.
        """
        T = root_pos.shape[0]

        # Apply overrides
        if root_height_override is not None:
            root_pos = root_pos.copy()
            root_pos[:, 2] = root_height_override
        if z_offset != 0:
            root_pos = root_pos.copy()
            root_pos[:, 2] += z_offset

        dof_29 = np.zeros((T, 29), dtype=np.float32)

        # Resample to model FPS if needed
        if src_fps is not None and abs(src_fps - self.model_fps) > 0.5:
            qpos_tmp = build_gt_qpos(root_pos, root_wxyz, dof_29)
            qpos_tmp = resample_qpos(qpos_tmp, src_fps, self.model_fps)
            root_pos = qpos_tmp[:, :3]
            root_wxyz = qpos_tmp[:, 3:7][:, [3, 0, 1, 2]]  # xyzw → wxyz
            dof_29 = qpos_tmp[:, 7:36]
            T_new = root_pos.shape[0]
            left_ee_pos = _resample_pos(left_ee_pos, T_new)
            left_ee_quat_wxyz = _resample_quat_wxyz(left_ee_quat_wxyz, T_new)
            right_ee_pos = _resample_pos(right_ee_pos, T_new)
            right_ee_quat_wxyz = _resample_quat_wxyz(right_ee_quat_wxyz, T_new)
            T = T_new

        # Compute model features from default standing pose (needed for EE override)
        global_motions, local_motions = arrays_to_mfm_features(
            root_pos,
            root_wxyz,
            dof_29,
            self._skeleton,
            self._global_motion_rep,
            self._local_motion_rep,
            self._motion_rep,
            self._device,
            xml_path=self._xml_path,
        )

        # Build EE override using canonical root from features
        ee_override_np = _build_ee_override(
            root_pos,
            root_wxyz,
            left_ee_pos,
            left_ee_quat_wxyz,
            right_ee_pos,
            right_ee_quat_wxyz,
            global_motions,
            self._global_motion_rep,
            self._device,
        )
        ee_override = torch.from_numpy(ee_override_np).float().to(self._device)

        # Align frame range to token boundaries
        start_frame = 0
        end_frame = (T // self.nfpt) * self.nfpt

        # Run chunked inference for each mode
        results = {}
        for mode in modes:
            predictions, chunk_kf_info = run_chunked_inference(
                inferencer=self._inferencer,
                models=self._models,
                global_motions=global_motions,
                local_motions=local_motions,
                start_frame=start_frame,
                end_frame=end_frame,
                chunk_tokens=max_chunk_tokens,
                mode=mode,
                ee_only_no_root=False,
                start_root_only=self._start_root_only,
                ee_override=ee_override,
                half_stride_blend=half_stride_blend,
            )

            pred_qpos = stitch_predictions(
                predictions,
                converter=self._converter,
                motion_rep=self._motion_rep,
                nfpt=self.nfpt,
                smooth=smooth,
            )

            results[mode] = {
                "qpos": pred_qpos.numpy() if pred_qpos is not None else None,
                "chunk_kf_info": chunk_kf_info,
            }

        return results


# -- EE override construction -------------------------------------------------------


def _build_ee_override(
    root_pos,
    root_wxyz,
    left_pos,
    left_quat_wxyz,
    right_pos,
    right_quat_wxyz,
    global_motions,
    global_motion_rep,
    device,
):
    """Build (1, T, 18) EE override in model RIC space.

    Matches the original infer_from_ee_positions exactly:
    - First-frame heading extracted via 2*arctan2(qy, qw) on wxyz quaternion
    - Canonical root Y from global_motion_rep.compute_root_pos_and_rot (per-frame)
    - Palm correction from retargeting matrices (NOT identity)
    """
    T = root_pos.shape[0]

    root_mfm_world = _MJ_TO_MODEL.apply(root_pos)

    # Canonical root from MFM features (per-frame, not constant)
    canon_root = (
        global_motion_rep.compute_root_pos_and_rot(
            global_motions.to(device), return_quat=False, return_angle=False
        )[0]
        .cpu()
        .numpy()
    )  # [1, T, 3] → squeeze batch
    if canon_root.ndim == 3:
        canon_root = canon_root[0]  # [T, 3]

    # First-frame heading from root quaternion in model space
    root_quat_mfm_f0 = (
        _MJ_TO_MODEL
        * Rotation.from_quat(root_wxyz[0, [1, 2, 3, 0]])
        * _MJ_TO_MODEL.inv()
    )
    q0_wxyz = root_quat_mfm_f0.as_quat()[[3, 0, 1, 2]]  # xyzw → wxyz
    init_heading = 2 * np.arctan2(q0_wxyz[2], q0_wxyz[0])
    h0_inv = Rotation.from_euler("Y", -init_heading)

    def _pos(ee_mj):
        ee_mfm = _MJ_TO_MODEL.apply(ee_mj)
        override = np.zeros((T, 3))
        for f in range(T):
            rel = ee_mfm[f].copy()
            rel[0] -= root_mfm_world[0, 0]
            rel[2] -= root_mfm_world[0, 2]
            canonical = h0_inv.apply(rel.reshape(1, 3))[0]
            canonical[1] += canon_root[f, 1]
            override[f] = canonical
        return override

    def _rot(ee_wxyz, palm_inv):
        rot6d = np.zeros((T, 6))
        for f in range(T):
            R_ext = Rotation.from_quat(ee_wxyz[f, [1, 2, 3, 0]])
            R_mfm = _MJ_TO_MODEL * R_ext * palm_inv * _MJ_TO_MODEL.inv()
            R_canon = h0_inv * R_mfm
            rot6d[f] = quaternion_to_cont6d_np(R_canon.as_quat()[[3, 0, 1, 2]][None])[0]
        return rot6d

    override = np.concatenate(
        [
            _pos(left_pos),
            _pos(right_pos),
            _rot(left_quat_wxyz, _R_PALM_LEFT_INV),
            _rot(right_quat_wxyz, _R_PALM_RIGHT_INV),
        ],
        axis=-1,
    ).astype(np.float32)
    return override[np.newaxis]


# -- Resampling helpers -------------------------------------------------------


def _resample_pos(pos, T_new):
    if pos.shape[0] == T_new:
        return pos

    t_old = np.linspace(0, 1, pos.shape[0])
    t_new = np.linspace(0, 1, T_new)
    return interp1d(t_old, pos, axis=0)(t_new).astype(np.float32)


def _resample_quat_wxyz(q, T_new):
    if q.shape[0] == T_new:
        return q

    q_xyzw = q[:, [1, 2, 3, 0]]
    for i in range(1, len(q_xyzw)):
        if np.dot(q_xyzw[i], q_xyzw[i - 1]) < 0:
            q_xyzw[i] = -q_xyzw[i]
    t_old = np.linspace(0, 1, q.shape[0])
    t_new = np.linspace(0, 1, T_new)
    out = Slerp(t_old, Rotation.from_quat(q_xyzw))(t_new).as_quat()
    return out[:, [3, 0, 1, 2]].astype(np.float32)
