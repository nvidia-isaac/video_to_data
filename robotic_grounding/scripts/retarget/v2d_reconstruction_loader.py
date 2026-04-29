# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load V2D reconstruction outputs into ManoSharpaData schema (MANO + object only).

Takes three explicit input paths rather than scanning a directory tree, so it
is agnostic to the reconstruction pipeline's internal layout:

  --mano_params_path   mano_params_*.npz produced by recover_mano_params
  --poses_dir          directory of per-frame object-to-camera pose JSONs
  --mesh_path          metric object mesh (.obj)
  --sequence_id        unique identifier for the output parquet partition
  --object_name        short label used for URDF lookup (default: sequence_id)

Example:
  python scripts/retarget/v2d_reconstruction_loader.py \\
    --mano_params_path /data/10_repro/10/hand_mesh/mano_params_moge.npz \\
    --poses_dir        /data/10_repro/10/poses_moge_smoothed \\
    --mesh_path        /data/10_repro/10/mesh_scaled.obj \\
    --sequence_id      v2d_reconstruction_10_moge \\
    --object_name      10 \\
    --save

Coordinate frame:
  MANO params and object poses are in OpenCV camera space:
    X right, Y down, Z forward (into scene).
  This loader rotates everything to a Z-up world frame:
    world X = cam X (right)
    world Y = cam Z (forward/depth)
    world Z = -cam Y (up)
  The camera position at frame 0 is used as the world origin.

MANO convention:
  DynHaMR stores pose_body as full axis-angle (not a deviation from hands_mean),
  so flat_hand_mean=True is correct. Using flat_hand_mean=False would cause manotorch
  to add hands_mean on top of the already-full pose, producing contorted hands.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from robotic_grounding.retarget import ASSETS_DIR, HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.dataset_loader_base import DatasetLoaderBase, SequenceInfo
from scipy.spatial.transform import Rotation

V2D_FPS = 25.0

# OpenCV camera (X right, Y down, Z forward) → Z-up world (X right, Y forward, Z up)
CAM_TO_WORLD = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=np.float64)

DEFAULT_OUTPUT_DIR = HUMAN_MOTION_DATA_DIR / "v2d_reconstruction" / "v2d_reconstruction_loaded"


@dataclass
class V2DReconstructionSequenceSource:
    """Explicit input paths — no assumptions about surrounding directory structure."""
    mano_params_path: Path
    poses_dir: Path
    mesh_path: Path


def _apply_cam_to_world_orient(global_orient: np.ndarray) -> np.ndarray:
    """Transform (B, T, 3) axis-angle rotations from camera to world frame."""
    B, T, _ = global_orient.shape
    flat = global_orient.reshape(-1, 3)
    R_cam = Rotation.from_rotvec(flat).as_matrix()       # (B*T, 3, 3)
    R_world = CAM_TO_WORLD[None] @ R_cam                 # (B*T, 3, 3)
    return Rotation.from_matrix(R_world).as_rotvec().reshape(B, T, 3).astype(np.float32)


def _apply_cam_to_world_trans(transl: np.ndarray) -> np.ndarray:
    """Transform (B, T, 3) translations from camera to world frame."""
    B, T, _ = transl.shape
    return (CAM_TO_WORLD @ transl.reshape(-1, 3).T).T.reshape(B, T, 3).astype(np.float32)


class V2DReconstructionLoader(DatasetLoaderBase):

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        object_name = args.object_name or args.sequence_id
        source = V2DReconstructionSequenceSource(
            mano_params_path = Path(args.mano_params_path),
            poses_dir        = Path(args.poses_dir),
            mesh_path        = Path(args.mesh_path),
        )
        return [SequenceInfo(
            sequence_id       = args.sequence_id,
            raw_motion_file   = str(args.mano_params_path),
            object_name       = object_name,
            object_body_names = [object_name],
            source            = source,
        )]

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        src: V2DReconstructionSequenceSource = sequence_info.source
        data = np.load(src.mano_params_path, allow_pickle=True)

        global_orient = data["global_orient"].astype(np.float32)   # (B, T, 3)
        transl        = data["transl"].astype(np.float32)           # (B, T, 3)
        hand_pose     = data["hand_pose"].astype(np.float32)        # (B, T, 45)
        betas         = data["betas"].astype(np.float32)            # (B, 10)
        is_right      = data["is_right"]                            # (B, T)
        rmsd_key      = next(k for k in ("vertex_rmsd", "procrustes_rmsd", "centroid_rmsd") if k in data)
        rmsd          = data[rmsd_key].astype(np.float32)          # (B, T)

        B, T = global_orient.shape[:2]

        go_world = _apply_cam_to_world_orient(global_orient)
        tr_world = _apply_cam_to_world_trans(transl)

        # Assign left/right by majority vote of is_right across frames
        hand_is_right = is_right.mean(axis=1) > 0.5  # (B,)
        right_idx = np.where(hand_is_right)[0]
        left_idx  = np.where(~hand_is_right)[0]

        def _pick(idx: np.ndarray, arr: np.ndarray, zero_shape: tuple) -> np.ndarray:
            return arr[idx[0]] if len(idx) > 0 else np.zeros((T,) + zero_shape, dtype=np.float32)

        def _pick_betas(idx: np.ndarray) -> np.ndarray:
            return betas[idx[0]] if len(idx) > 0 else np.zeros(10, dtype=np.float32)

        def _pick_rmsd(idx: np.ndarray) -> np.ndarray:
            return rmsd[idx[0]] if len(idx) > 0 else np.zeros(T, dtype=np.float32)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32, device=device)

        return {
            "H":                  T,
            "right_global_orient":t(_pick(right_idx, go_world,  (3,))),
            "right_finger_pose":  t(_pick(right_idx, hand_pose, (45,))),
            "right_trans":        t(_pick(right_idx, tr_world,  (3,))),
            "right_betas":        t(_pick_betas(right_idx)),
            "right_fitting_err":  t(_pick_rmsd(right_idx)),
            "left_global_orient": t(_pick(left_idx,  go_world,  (3,))),
            "left_finger_pose":   t(_pick(left_idx,  hand_pose, (45,))),
            "left_trans":         t(_pick(left_idx,  tr_world,  (3,))),
            "left_betas":         t(_pick_betas(left_idx)),
            "left_fitting_err":   t(_pick_rmsd(left_idx)),
        }

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        src: V2DReconstructionSequenceSource = sequence_info.source
        pose_files = sorted(src.poses_dir.glob("*.json"))
        T = len(pose_files)

        poses = np.zeros((T, 4, 4), dtype=np.float64)
        poses[:, 3, 3] = 1.0

        for i, p in enumerate(pose_files):
            with open(p) as f:
                td = json.load(f)
            w, x, y, z = td["rotation"]                           # stored as wxyz
            R_cam = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy: xyzw
            t_cam = np.array(td["translation"])

            poses[i, :3, :3] = CAM_TO_WORLD @ R_cam
            poses[i, :3,  3] = CAM_TO_WORLD @ t_cam

        root_pos = poses[:, :3, 3].astype(np.float32)
        root_aa  = Rotation.from_matrix(poses[:, :3, :3]).as_rotvec().astype(np.float32)

        return {sequence_info.object_name: (poses, root_pos, root_aa, None)}

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple:
        src: V2DReconstructionSequenceSource = sequence_info.source
        mesh = trimesh.load(str(src.mesh_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.array(mesh.faces),    dtype=torch.int64,   device=device)

        pts, face_idxs = trimesh.sample.sample_surface(mesh, 2000)
        nrm = mesh.face_normals[face_idxs]

        name = sequence_info.object_name
        return (
            {name: mesh},
            {name: verts},
            {name: faces},
            {name: torch.tensor(pts.astype(np.float32),  dtype=torch.float32, device=device)},
            {name: torch.tensor(nrm.astype(np.float32),  dtype=torch.float32, device=device)},
            True,
        )

    def get_mano_kwargs(self) -> dict[str, Any]:
        return {"flat_hand_mean": True, "center_idx": None}

    def get_fps(self) -> float:
        return V2D_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        src: V2DReconstructionSequenceSource = sequence_info.source
        return [str(src.mesh_path)]

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        urdf_dir = ASSETS_DIR / "urdfs" / "v2d_reconstruction"
        return [str(urdf_dir / f"{sequence_info.object_name}_rigid.urdf")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load V2D reconstruction outputs into ManoSharpaData parquet schema."
    )
    parser.add_argument(
        "--mano_params_path", type=Path, required=True,
        help="Path to mano_params_*.npz (output of recover_mano_params).",
    )
    parser.add_argument(
        "--poses_dir", type=Path, required=True,
        help="Directory of per-frame object-to-camera pose JSONs (e.g. poses_moge_smoothed/).",
    )
    parser.add_argument(
        "--mesh_path", type=Path, required=True,
        help="Path to the scaled object mesh (.obj).",
    )
    parser.add_argument(
        "--object_name", type=str, default=None,
        help="Short label for URDF lookup. Defaults to --sequence_id.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    DatasetLoaderBase.add_filter_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not args.sequence_id:
        raise ValueError("--sequence_id is required")
    loader = V2DReconstructionLoader()
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
