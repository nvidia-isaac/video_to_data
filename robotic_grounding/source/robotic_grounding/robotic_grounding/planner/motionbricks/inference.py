# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MotionBricks whole-body planner: end-effector targets -> full-body qpos.

``MotionInferenceAgent.infer_from_ee_positions`` takes dual-wrist EE targets in
MuJoCo z-up world frame and returns a chunked autoregressive whole-body plan as
qpos with layout ``[pos(3), xyzw_quat(4), joints(29)]``. Inference runs on three
ONNX graphs (root, pose, decode) plus numpy seed/canonicalization/qpos helpers —
numpy + scipy + onnxruntime only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robotic_grounding.planner.motionbricks.canonicalization import (
    _canonicalize_ee_targets,
    _canonicalize_heading_transforms,
)
from robotic_grounding.planner.motionbricks.kinematics import (
    DEFAULT_SEED_XML,
    HAND_ROOT_TO_WRIST_OFFSET_LOCAL_LEFT,
    HAND_ROOT_TO_WRIST_OFFSET_LOCAL_RIGHT,
    apply_hand_root_to_wrist_offset,
    build_seed_qpos,
    features_to_qpos_np,
    qpos_to_body_world,
)
from robotic_grounding.planner.motionbricks.runtime import (
    MotionBricksOnnxEngine,
    blend_two_passes,
)
from robotic_grounding.planner.motionbricks.smoothing import (
    _chunk_boundary_centers,
    smooth_qpos_at_boundaries,
    smooth_qpos_global,
)

# ONNX graphs + kinematics sidecar ship alongside this package.
_DEFAULT_ONNX_DIR = str(Path(__file__).parent / "assets" / "models")


class MotionInferenceAgent:
    """Whole-body motion inference from end-effector targets.

    Inputs are MuJoCo z-up world frame; output qpos layout is
    ``[pos(3), xyzw_quat(4), joints(29)]``.
    """

    def __init__(self, device: str = "cpu", onnx_dir: str | None = None) -> None:
        """Load the root/pose/decode ONNX graphs and the kinematics sidecar.

        ``device`` is accepted for call-site parity; inference runs on the CPU
        execution provider.
        """
        del device
        self._onnx_dir = onnx_dir or _DEFAULT_ONNX_DIR
        self._engine = MotionBricksOnnxEngine(self._onnx_dir)

    @property
    def nfpt(self) -> int:
        """Number of motion frames represented by each model token."""
        return int(self._engine.nfpt)

    @property
    def model_fps(self) -> int:
        """Frame rate the model operates at (resampling is handled upstream)."""
        return 25

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens the model can generate in one sequence."""
        return int(self._engine.max_tokens)

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
        overlap_tokens: int = 3,
        modes: tuple[str, ...] = ("autoregressive",),
        smooth: bool = True,
        half_stride_blend: bool = True,
        left_hand_root_offset_local: tuple[float, float, float] | None = (
            HAND_ROOT_TO_WRIST_OFFSET_LOCAL_LEFT
        ),
        right_hand_root_offset_local: tuple[float, float, float] | None = (
            HAND_ROOT_TO_WRIST_OFFSET_LOCAL_RIGHT
        ),
        fix_lower_body: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Plan whole-body motion from dual-wrist EE targets.

        Returns ``{"autoregressive": {"qpos": (T, 36), "chunk_kf_info": [...]}}``
        with qpos in MuJoCo xyzw-quaternion layout.
        """
        del src_fps, modes  # accepted for call-site parity

        if root_height_override is not None:
            root_pos = root_pos.copy()
            root_pos[:, 2] = root_height_override
        if z_offset != 0.0:
            root_pos = root_pos.copy()
            root_pos[:, 2] += z_offset

        # Inputs track the hand-root body (palm_link); the model expects
        # wrist_yaw_link positions. Shift along the forearm before canonicalizing.
        if left_hand_root_offset_local is not None:
            left_ee_pos = apply_hand_root_to_wrist_offset(
                left_ee_pos, left_ee_quat_wxyz, left_hand_root_offset_local
            )
        if right_hand_root_offset_local is not None:
            right_ee_pos = apply_hand_root_to_wrist_offset(
                right_ee_pos, right_ee_quat_wxyz, right_hand_root_offset_local
            )

        T = root_pos.shape[0]

        # FK a static-pose qpos so the canonicalization sees the coordinate
        # convention the model was trained in.
        seed_qpos, seed_joint_names = build_seed_qpos(
            num_frames=T, root_height=float(root_pos[0, 2])
        )
        body_pos_w, body_wxyz_w = qpos_to_body_world(
            seed_qpos, seed_joint_names, DEFAULT_SEED_XML
        )

        gt_joint_transforms, gt_root_xy = _canonicalize_heading_transforms(
            body_pos_w=body_pos_w,
            body_wxyz_w=body_wxyz_w,
            root_pos_w=seed_qpos[:, :3],
            root_wxyz_w=seed_qpos[:, 3:7],
        )
        root_ee_transforms, pose_ee_transforms, _marker = _canonicalize_ee_targets(
            left_pos_w=left_ee_pos,
            left_wxyz_w=left_ee_quat_wxyz,
            right_pos_w=right_ee_pos,
            right_wxyz_w=right_ee_quat_wxyz,
            root_pos_w=seed_qpos[:, :3],
            root_wxyz_w=seed_qpos[:, 3:7],
        )

        root_ee_input = (
            root_ee_transforms if self._engine.root_target_ee else pose_ee_transforms
        )
        derive_pose = self._engine.derive_pose_ee_from_pred_root
        pose_ee_input = None if derive_pose else pose_ee_transforms

        nfpt = self._engine.nfpt
        chunk_frames = int(max_chunk_tokens) * nfpt
        half_stride = chunk_frames // 2

        def _run_pass(start_offset: int) -> dict[str, Any]:
            ee_in = root_ee_input[start_offset:]
            pose_ee_in = (
                pose_ee_input[start_offset:] if pose_ee_input is not None else None
            )
            static_pose = gt_joint_transforms[start_offset:] if fix_lower_body else None
            return self._engine.run_chunked_autoregressive_inference(
                root_ee_input=ee_in,
                seed_root=gt_root_xy[start_offset:],
                seed_pose=gt_joint_transforms[start_offset:],
                pose_ee_input=pose_ee_in,
                chunk_tokens=int(max_chunk_tokens),
                overlap_tokens=int(overlap_tokens),
                static_pose_shared=static_pose,
            )

        # Optional half-stride second pass, averaged to soften chunk seams.
        result_a = _run_pass(0)
        if (
            half_stride_blend
            and half_stride >= nfpt
            and root_ee_input.shape[0] - half_stride >= nfpt
        ):
            result_b = _run_pass(half_stride)
            result = blend_two_passes(
                result_a,
                result_b,
                offset_b=half_stride,
                total_frames=root_ee_input.shape[0],
            )
        else:
            result = result_a

        qpos_36 = features_to_qpos_np(
            result["pred_joints"],
            result["pred_root"],
            self._engine.kin,
            self._engine.body_reorder,
        )

        if smooth:
            qpos_36 = smooth_qpos_global(qpos_36, nfpt=nfpt)
            boundaries = _chunk_boundary_centers(result.get("chunk_infos") or [])
            if boundaries:
                qpos_36 = smooth_qpos_at_boundaries(qpos_36, boundaries, nfpt=nfpt)

        qpos_xyzw = qpos_36.copy()
        qpos_xyzw[:, 3:7] = qpos_36[:, [4, 5, 6, 3]]  # root quat wxyz -> xyzw

        return {
            "autoregressive": {
                "qpos": qpos_xyzw.astype(np.float32),
                "chunk_kf_info": result.get("chunk_infos", []),
            }
        }
