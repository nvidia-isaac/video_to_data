# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load OakInk2 dataset into ManoSharpaData schema (MANO + object only, no robot).

OakInk2 layout:
  oakink_dir/
    anno_preview/   -> {sequence}.pkl  (keys: raw_mano, obj_transf, obj_list, cam_extr)
    object_repair/align_ds/{object_id}/model.obj
    object_raw/align_ds/{object_id}/model.obj

Poses are stored in OakInk2's y-up OptiTrack world frame. This loader applies a
world -> z-up coordinate transform (x=x, y=-z, z=y) so that saved Parquet data
uses the same z-up convention as ARCTIC and TACO.

Runs stage 1 of the two-stage pipeline:
  1. python scripts/retarget/oakink2_loader.py --save   -> oakink2_loaded/
  2. python scripts/retarget/oakink2_to_sharpa.py --save -> oakink2_processed/
"""

import argparse
import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.dataset_loader_base import (
    DatasetLoaderBase,
    SequenceInfo,
    load_meshes_to_device,
    poses_to_root_position_and_axis_angle,
)
from scipy.spatial.transform import Rotation

logging.getLogger().setLevel(logging.ERROR)
# manotorch uses torch.cross without dim arg (deprecated in newer PyTorch); suppress.
warnings.filterwarnings("ignore", category=UserWarning, module="manotorch")

DEFAULT_OAKINK_DIR = HUMAN_MOTION_DATA_DIR / "oakink2"
LOADED_SAVE_DIR = HUMAN_MOTION_DATA_DIR / "oakink2_loaded"
OAKINK2_FPS = 30.0

# Rotation: OakInk2/OptiTrack y-up world -> z-up convention used by ARCTIC/TACO
#   x_new =  x_old
#   y_new = -z_old
#   z_new =  y_old
_WORLD_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
_WORLD_TO_ZUP_T4 = np.eye(4, dtype=np.float64)
_WORLD_TO_ZUP_T4[:3, :3] = _WORLD_TO_ZUP


@dataclass
class OakInk2SequenceSource:
    """Dataset-specific metadata stored in SequenceInfo.source."""

    seq_pkl: Path
    oakink_dir: Path
    use_object_raw: bool
    object_ids: list[str]  # from seq_data["obj_list"]


# ---------------------------------------------------------------------------
# Coordinate-frame helpers
# ---------------------------------------------------------------------------


def _quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert OakInk2 quaternions [w,x,y,z] (...,4) to rotation vectors (...,3)."""
    return (
        Rotation.from_quat(np.asarray(quat_wxyz, dtype=np.float32), scalar_first=True)
        .as_rotvec()
        .astype(np.float32)
    )


def _apply_zup_to_trans(trans: np.ndarray) -> np.ndarray:
    """Rotate (N,3) or (3,) translations from y-up world to z-up."""
    single = trans.ndim == 1
    t = trans.reshape(-1, 3).astype(np.float64)
    t_new = (_WORLD_TO_ZUP @ t.T).T.astype(np.float32)
    return t_new[0] if single else t_new


def _apply_zup_to_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Transform (N,3) rotation vectors: R_new = R_zup @ R_old.

    finger_pose (relative joint rotations) is NOT passed here — it is
    frame-independent and does not change when the world frame changes.
    """
    R_old = Rotation.from_rotvec(rotvec).as_matrix()  # (N,3,3)
    R_new = _WORLD_TO_ZUP @ R_old  # (3,3) @ (N,3,3) -> (N,3,3)
    return Rotation.from_matrix(R_new).as_rotvec().astype(np.float32)


def _apply_zup_to_poses(poses: np.ndarray) -> np.ndarray:
    """Apply world->z-up transform to (N,4,4) pose matrices."""
    # _WORLD_TO_ZUP_T4 @ poses[i]: rotation applied to both rotation and translation
    return np.einsum("ij,njk->nik", _WORLD_TO_ZUP_T4, poses.astype(np.float64))


# ---------------------------------------------------------------------------
# OakInk2 data helpers
# ---------------------------------------------------------------------------


def _load_seq_pkl(seq_pkl: Path) -> dict:
    with seq_pkl.open("rb") as f:
        return pickle.load(f)


def _extract_hand_from_entry(
    entry: dict, prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (global_orient_rotvec, finger_pose_rotvec, trans, betas) for one hand frame.

    prefix is "rh" (right) or "lh" (left).
    pose_coeffs: (16,4) quaternions [w,x,y,z] — wrist + 15 finger joints.
    """
    pose_key = f"{prefix}__pose_coeffs"
    tsl_key = f"{prefix}__tsl"
    betas_key = f"{prefix}__betas"

    if pose_key not in entry:
        raise KeyError(f"Missing key '{pose_key}'")

    pose_quat = np.asarray(entry[pose_key], dtype=np.float32).squeeze()
    if pose_quat.shape != (16, 4):
        raise ValueError(f"{pose_key} expected shape (16,4), got {pose_quat.shape}")

    pose_rotvec = _quat_wxyz_to_rotvec(pose_quat)  # (16,3)
    global_orient = pose_rotvec[0]  # (3,)  wrist orientation
    finger_pose = pose_rotvec[1:].reshape(45)  # (45,) 15 joints × 3

    trans = np.asarray(entry[tsl_key], dtype=np.float32).reshape(3)
    betas = np.asarray(entry[betas_key], dtype=np.float32).reshape(-1)[:10]
    return global_orient, finger_pose, trans, betas


def _resolve_mesh_path(oakink_dir: Path, object_id: str, use_object_raw: bool) -> Path:
    """Resolve object mesh path; prefers object_repair, falls back to object_raw."""
    primary = "object_raw" if use_object_raw else "object_repair"
    path = oakink_dir / primary / "align_ds" / object_id / "model.obj"
    if path.exists():
        return path
    fallback = "object_repair" if use_object_raw else "object_raw"
    path2 = oakink_dir / fallback / "align_ds" / object_id / "model.obj"
    if path2.exists():
        return path2
    raise FileNotFoundError(
        f"No mesh for '{object_id}' in {oakink_dir} (checked {primary} and {fallback})"
    )


def _extract_valid_frames(
    seq_data: dict, object_ids: list[str]
) -> tuple[
    list, list, list, list, list, list, list, np.ndarray | None, np.ndarray | None
]:
    """Iterate raw_mano, convert quaternions, filter to frames with object poses.

    Returns lists for right and left: (global_orient, finger_pose, trans),
    plus right_betas and left_betas (taken from first valid frame).
    """
    raw_mano = seq_data["raw_mano"]
    obj_transf = seq_data["obj_transf"]

    rg_list, rp_list, rt_list = [], [], []
    lg_list, lp_list, lt_list = [], [], []
    right_betas: np.ndarray | None = None
    left_betas: np.ndarray | None = None
    valid_frame_ids = []

    for fid in sorted(raw_mano.keys()):
        entry = raw_mano[fid]
        try:
            rg, rp, rt, rb = _extract_hand_from_entry(entry, "rh")
            lg, lp, lt, lb = _extract_hand_from_entry(entry, "lh")
        except (KeyError, ValueError):
            continue

        # Require object pose data for every object in this sequence
        if not all(oid in obj_transf and fid in obj_transf[oid] for oid in object_ids):
            continue

        valid_frame_ids.append(fid)
        rg_list.append(rg)
        rp_list.append(rp)
        rt_list.append(rt)
        lg_list.append(lg)
        lp_list.append(lp)
        lt_list.append(lt)
        if right_betas is None:
            right_betas = rb
        if left_betas is None:
            left_betas = lb

    return (
        valid_frame_ids,
        rg_list,
        rp_list,
        rt_list,
        lg_list,
        lp_list,
        lt_list,
        right_betas,
        left_betas,
    )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


class OakInk2DatasetLoader(DatasetLoaderBase):
    """OakInk2 dataset loader."""

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """Discover all anno_preview .pkl sequences."""
        oakink_dir = Path(args.oakink_dir)
        use_object_raw = getattr(args, "use_object_raw", False)
        seq_name_filter = getattr(args, "seq_name", None)

        anno_dir = oakink_dir / "anno_preview"
        if not anno_dir.is_dir():
            raise FileNotFoundError(f"anno_preview not found: {anno_dir}")

        pkls = sorted(anno_dir.glob("*.pkl"))
        if seq_name_filter:
            pkls = [p for p in pkls if seq_name_filter in p.stem]

        out = []
        for pkl_path in pkls:
            try:
                seq_data = _load_seq_pkl(pkl_path)
            except Exception as e:
                print(f"Skipping {pkl_path.name}: failed to load pkl: {e}")
                continue

            if not {"raw_mano", "obj_transf", "obj_list"}.issubset(seq_data.keys()):
                continue

            object_ids = list(seq_data["obj_list"])
            if not object_ids or not seq_data["raw_mano"]:
                continue

            sequence_id = pkl_path.stem
            out.append(
                SequenceInfo(
                    sequence_id=sequence_id,
                    raw_motion_file=pkl_path.stem,
                    object_name="+".join(object_ids),
                    object_body_names=list(object_ids),
                    source=OakInk2SequenceSource(
                        seq_pkl=pkl_path,
                        oakink_dir=oakink_dir,
                        use_object_raw=use_object_raw,
                        object_ids=object_ids,
                    ),
                )
            )
        return out

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO parameters from OakInk2 pkl; apply world->z-up transform."""
        src: OakInk2SequenceSource = sequence_info.source
        seq_data = _load_seq_pkl(src.seq_pkl)

        (
            valid_frame_ids,
            rg_list,
            rp_list,
            rt_list,
            lg_list,
            lp_list,
            lt_list,
            right_betas,
            left_betas,
        ) = _extract_valid_frames(seq_data, src.object_ids)

        if not valid_frame_ids:
            raise ValueError(f"No valid frames in {src.seq_pkl.name}")

        # Cache valid_frame_ids so load_object_data uses the same filtered set
        if not hasattr(self, "_valid_frame_ids_cache"):
            self._valid_frame_ids_cache: dict[str, list] = {}
        self._valid_frame_ids_cache[sequence_info.sequence_id] = valid_frame_ids

        # Apply world->z-up transform to translations and global orientations.
        # finger_pose is relative (joint-to-joint) so it is unaffected.
        right_global_orient = _apply_zup_to_rotvec(np.stack(rg_list))  # (N,3)
        right_finger_pose = np.stack(rp_list).astype(np.float32)  # (N,45)
        right_trans = _apply_zup_to_trans(np.stack(rt_list))  # (N,3)

        left_global_orient = _apply_zup_to_rotvec(np.stack(lg_list))
        left_finger_pose = np.stack(lp_list).astype(np.float32)
        left_trans = _apply_zup_to_trans(np.stack(lt_list))

        H = len(valid_frame_ids)
        return {
            "right_global_orient": torch.from_numpy(right_global_orient).to(device),
            "right_finger_pose": torch.from_numpy(right_finger_pose).to(device),
            "right_trans": torch.from_numpy(right_trans).to(device),
            "right_betas": torch.from_numpy(right_betas).to(device),
            "right_fitting_err": torch.zeros(H, device=device),
            "left_global_orient": torch.from_numpy(left_global_orient).to(device),
            "left_finger_pose": torch.from_numpy(left_finger_pose).to(device),
            "left_trans": torch.from_numpy(left_trans).to(device),
            "left_betas": torch.from_numpy(left_betas).to(device),
            "left_fitting_err": torch.zeros(H, device=device),
            "H": H,
        }

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Load object pose data; apply world->z-up transform. Returns rigid objects (no articulation)."""
        src: OakInk2SequenceSource = sequence_info.source
        seq_data = _load_seq_pkl(src.seq_pkl)
        obj_transf = seq_data["obj_transf"]

        valid_frame_ids = self._valid_frame_ids_cache.get(sequence_info.sequence_id)
        if valid_frame_ids is None:
            raise RuntimeError(
                "load_mano_data must be called before load_object_data "
                f"(no cached frame ids for '{sequence_info.sequence_id}')"
            )

        result: dict[str, Any] = {}
        for oid in src.object_ids:
            poses_raw = np.array(
                [
                    np.asarray(obj_transf[oid][fid], dtype=np.float64)
                    for fid in valid_frame_ids
                ]
            )  # (N,4,4)
            poses_zup = _apply_zup_to_poses(poses_raw)  # (N,4,4)
            root_pos, root_aa = poses_to_root_position_and_axis_angle(poses_zup)
            result[oid] = (poses_zup, root_pos, root_aa, None)  # None = no articulation

        return result

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], bool]:
        """Load OakInk2 object meshes (model.obj, already in meters)."""
        src: OakInk2SequenceSource = sequence_info.source
        mesh_paths: dict[str, str] = {}
        for oid in src.object_ids:
            try:
                mesh_paths[oid] = str(
                    _resolve_mesh_path(src.oakink_dir, oid, src.use_object_raw)
                )
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        return load_meshes_to_device(mesh_paths, device)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """OakInk2 uses flat_hand_mean=True and centers at wrist (center_idx=0)."""
        return {"flat_hand_mean": True, "center_idx": 0}

    def get_fps(self) -> float:
        """Return the frames-per-second for OakInk2 sequences."""
        return OAKINK2_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return mesh paths for all objects in the sequence (one per object_id)."""
        src: OakInk2SequenceSource = sequence_info.source
        paths = []
        for oid in src.object_ids:
            try:
                paths.append(
                    str(_resolve_mesh_path(src.oakink_dir, oid, src.use_object_raw))
                )
            except FileNotFoundError:
                paths.append("")
        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the OakInk2 loader script."""
    parser = argparse.ArgumentParser(
        description="Load OakInk2 sequences into ManoSharpaData schema (MANO + object only)."
    )
    parser.add_argument(
        "--oakink_dir",
        type=Path,
        default=DEFAULT_OAKINK_DIR,
        help="Path to OakInk-v2-hub root directory.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "--use_object_raw",
        action="store_true",
        default=False,
        help="Use object_raw instead of object_repair meshes.",
    )
    parser.add_argument("--mano_to_robot_scale", type=float, default=1.2)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=LOADED_SAVE_DIR,
        help="Parent directory for Parquet output.",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default=None,
        help="Process only sequences whose filename contains this substring.",
    )
    parser.add_argument(
        "--list_sequences",
        action="store_true",
        default=False,
        help="List available sequences and exit.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the OakInk2 loader: list sequences or process and save ManoSharpaData Parquet files."""
    if args.list_sequences:
        loader = OakInk2DatasetLoader()
        sequences = loader.list_sequences(args)
        print(f"Found {len(sequences)} sequences in {args.oakink_dir / 'anno_preview'}")
        for i, s in enumerate(sequences, start=1):
            print(f"{i:3d}. {s.sequence_id}")
        return

    loader = OakInk2DatasetLoader()
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
