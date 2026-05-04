# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load V2D reconstruction outputs into ManoSharpaData schema (MANO + object only).

Two input formats are supported:

  (A) Legacy camera-space format  --format camera  [default when --mano_params_path given]
      --mano_params_path   mano_params_*.npz produced by recover_mano_params
      --poses_dir          directory of per-frame object-to-camera pose JSONs
      --mesh_path          metric object mesh (.obj)

  (B) World-results format  --format world  [default when --world_results_path given]
      --world_results_path world_results_aligned.npz (from align_world_results)
      --mesh_path          metric object mesh (.obj)
      Hand and object are placed in DynHaMR normalized units (trans unscaled;
      object mesh/poses divided by world_scale) so MANO verts and trans share
      the same coordinate frame.

Both formats rotate everything to a Z-up world frame before saving:
  world X = cam X (right)
  world Y = cam Z (forward/depth)
  world Z = -cam Y (up)

MANO convention:
  DynHaMR stores pose_body as full axis-angle (not a deviation from hands_mean),
  so flat_hand_mean=True is correct.

Examples:
  # Format A (legacy)
  python scripts/retarget/v2d_reconstruction_loader.py \\
    --mano_params_path /data/10_repro/10/hand_mesh/mano_params_moge.npz \\
    --poses_dir        /data/10_repro/10/poses_moge_smoothed \\
    --mesh_path        /data/10_repro/10/mesh_scaled.obj \\
    --sequence_id      v2d_10_moge_legacy --save

  # Format B (world results)
  python scripts/retarget/v2d_reconstruction_loader.py \\
    --world_results_path /data/10_repro/10/hand_mesh/world_results_aligned_moge.npz \\
    --mesh_path          /data/10_repro/10/mesh_scaled.obj \\
    --sequence_id        v2d_10_moge --save
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


# ---------------------------------------------------------------------------
# Format B: world_results_aligned.npz loader
# ---------------------------------------------------------------------------

@dataclass
class V2DWorldResultsSequenceSource:
    """Inputs for the world-results format."""
    world_results_path: Path
    mesh_path: Path


class V2DWorldResultsLoader(DatasetLoaderBase):
    """Load from world_results_aligned.npz + object mesh.

    Everything is kept in DynHaMR normalized units (same units as MANO verts_local
    and trans).  DynHaMR FK: v_metric = (verts_local + trans) * world_scale.
    Downstream MANO computes verts_local + transl, so transl must match verts_local
    scale — we pass trans unscaled.  Object poses (metric from FoundationPose) and
    mesh vertices are divided by world_scale to bring them into the same space.
    Both hand and object are rotated to Z-up convention via CAM_TO_WORLD.
    """

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        object_name = args.object_name or args.sequence_id
        source = V2DWorldResultsSequenceSource(
            world_results_path = Path(args.world_results_path),
            mesh_path          = Path(args.mesh_path),
        )
        return [SequenceInfo(
            sequence_id       = args.sequence_id,
            raw_motion_file   = str(args.world_results_path),
            object_name       = object_name,
            object_body_names = [object_name],
            source            = source,
        )]

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        src: V2DWorldResultsSequenceSource = sequence_info.source
        data = np.load(src.world_results_path, allow_pickle=True)

        root_orient = data['root_orient'].astype(np.float32)          # (B, T, 3)
        pose_body   = data['pose_body'  ].astype(np.float32)          # (B, T, 15, 3) or (B, T, 45)
        betas       = data['betas'      ].astype(np.float32)          # (B, 10)
        is_right    = data['is_right']                                  # (B, T)

        trans_key = 'trans_aligned' if 'trans_aligned' in data.files else 'trans'
        trans     = data[trans_key].astype(np.float32)                 # (B, T, 3) DynHaMR units

        B, T = root_orient.shape[:2]
        pose_body = pose_body.reshape(B, T, 45)

        # Pass trans in DynHaMR normalized units (same units as MANO verts_local).
        # DynHaMR FK: v_metric = (verts_local + trans) * world_scale.
        # Downstream MANO computes verts_local + transl, so transl must match verts_local scale.
        # Object mesh/poses are divided by world_scale in load_object_data/load_object_meshes
        # so that hand and object share a consistent coordinate frame.

        # Rotate from DynHaMR world frame (≈ OpenCV camera: Y-down, Z-forward) to Z-up.
        go_world = _apply_cam_to_world_orient(root_orient)
        tr_world = _apply_cam_to_world_trans(trans)

        hand_is_right = is_right.mean(axis=1) > 0.5
        right_idx = np.where(hand_is_right)[0]
        left_idx  = np.where(~hand_is_right)[0]

        def _pick(idx: np.ndarray, arr: np.ndarray, zero_shape: tuple) -> np.ndarray:
            return arr[idx[0]] if len(idx) > 0 else np.zeros((T,) + zero_shape, dtype=np.float32)

        def _pick_betas(idx: np.ndarray) -> np.ndarray:
            return betas[idx[0]] if len(idx) > 0 else np.zeros(10, dtype=np.float32)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32, device=device)

        return {
            "H":                  T,
            "right_global_orient":t(_pick(right_idx, go_world,  (3,))),
            "right_finger_pose":  t(_pick(right_idx, pose_body, (45,))),
            "right_trans":        t(_pick(right_idx, tr_world,  (3,))),
            "right_betas":        t(_pick_betas(right_idx)),
            "right_fitting_err":  t(np.zeros(T, dtype=np.float32)),
            "left_global_orient": t(_pick(left_idx,  go_world,  (3,))),
            "left_finger_pose":   t(_pick(left_idx,  pose_body, (45,))),
            "left_trans":         t(_pick(left_idx,  tr_world,  (3,))),
            "left_betas":         t(_pick_betas(left_idx)),
            "left_fitting_err":   t(np.zeros(T, dtype=np.float32)),
        }

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        src: V2DWorldResultsSequenceSource = sequence_info.source
        data = np.load(src.world_results_path, allow_pickle=True)

        if 'object_pose_world' not in data.files:
            raise KeyError(
                "object_pose_world not found in world_results_aligned.npz. "
                "Re-run align_world_results with --object_poses_dir."
            )

        world_scale = float(data['world_scale'].flat[0]) if 'world_scale' in data else 1.0
        poses_metric = data['object_pose_world'].astype(np.float64)    # (T, 4, 4) metric

        # Convert translation from metric → DynHaMR normalized (same units as MANO verts + trans).
        # Rotation is dimensionless — no scaling needed.
        T = len(poses_metric)
        poses = np.zeros((T, 4, 4), dtype=np.float64)
        poses[:, 3, 3] = 1.0
        poses[:, :3, :3] = CAM_TO_WORLD[None] @ poses_metric[:, :3, :3]
        poses[:, :3,  3] = (CAM_TO_WORLD @ (poses_metric[:, :3, 3] / world_scale).T).T

        root_pos = poses[:, :3, 3].astype(np.float32)
        root_aa  = Rotation.from_matrix(poses[:, :3, :3]).as_rotvec().astype(np.float32)

        return {sequence_info.object_name: (poses, root_pos, root_aa, None)}

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple:
        src: V2DWorldResultsSequenceSource = sequence_info.source
        data = np.load(src.world_results_path, allow_pickle=True)
        world_scale = float(data['world_scale'].flat[0]) if 'world_scale' in data else 1.0

        mesh = trimesh.load(str(src.mesh_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Scale mesh from metric → DynHaMR normalized to match hand coordinate frame.
        mesh.vertices /= world_scale

        verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.array(mesh.faces),    dtype=torch.int64,   device=device)

        pts, face_idxs = trimesh.sample.sample_surface(mesh, 2000)
        nrm = mesh.face_normals[face_idxs]

        name = sequence_info.object_name
        return (
            {name: mesh},
            {name: verts},
            {name: faces},
            {name: torch.tensor(pts.astype(np.float32), dtype=torch.float32, device=device)},
            {name: torch.tensor(nrm.astype(np.float32), dtype=torch.float32, device=device)},
            True,
        )

    def get_mano_kwargs(self) -> dict[str, Any]:
        return {"flat_hand_mean": True, "center_idx": None}

    def get_fps(self) -> float:
        return V2D_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        src: V2DWorldResultsSequenceSource = sequence_info.source
        return [str(src.mesh_path)]

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        urdf_dir = ASSETS_DIR / "urdfs" / "v2d_reconstruction"
        return [str(urdf_dir / f"{sequence_info.object_name}_rigid.urdf")]


# ---------------------------------------------------------------------------
# CLI — supports both formats
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load V2D reconstruction outputs into ManoSharpaData parquet schema."
    )
    # Format A (legacy camera-space)
    parser.add_argument(
        "--mano_params_path", type=Path, default=None,
        help="[Format A] Path to mano_params_*.npz (recover_mano_params output).",
    )
    parser.add_argument(
        "--poses_dir", type=Path, default=None,
        help="[Format A] Directory of per-frame object-to-camera pose JSONs.",
    )
    # Format B (world results)
    parser.add_argument(
        "--world_results_path", type=Path, default=None,
        help="[Format B] Path to world_results_aligned.npz (align_world_results output).",
    )
    # Shared
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

    if args.world_results_path is not None:
        loader = V2DWorldResultsLoader()
    elif args.mano_params_path is not None:
        if args.poses_dir is None:
            raise ValueError("--poses_dir is required with --mano_params_path")
        loader = V2DReconstructionLoader()
    else:
        raise ValueError("Provide either --world_results_path or --mano_params_path")

    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
