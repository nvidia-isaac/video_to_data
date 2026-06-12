# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Hot3D dataset into ManoSharpaData schema (MANO + object only, no robot).

Hot3D layout:
  hot3d_dir/
    {seq_name}/                     e.g. P0001_8d136980/
      mano_hand_pose_trajectory.jsonl   per-timestamp MANO poses (PCA format)
      dynamic_objects.csv               per-timestamp object world poses
      metadata.json                     object_uids, object_names, participant_id
      headset_trajectory.csv            headset world pose per timestamp
      masks/                            quality masks (optional)
  assets/meshes/hot3d/
      {object_uid}.glb                  object mesh files (meters)

Coordinate frame:
  Aria world frames are already Z-UP (gravity_z = -9.81).
  Quest3 world frames are Y-UP and are rotated to Z-UP via QUEST3_WORLD_TO_ZUP.

  After gravity alignment, the yaw (rotation around Z) is arbitrary per session.
  This loader also applies a per-sequence yaw normalization so that all scenes
  start with the headset's initial forward direction pointing toward +Y.  The
  normalization rotation is read from the first frame of headset_trajectory.csv.

MANO format:
  Poses are stored as 15 PCA coefficients (use_pca=True, num_pca_comps=15).
  This loader expands them to full 45-DOF rotation vectors before passing
  to our MANO wrapper (which expects use_pca=False, 45-DOF finger pose).
  Hand index 0 = left, hand index 1 = right.

Timestamp handling:
  Each "real" frame at ~30 Hz has 2-3 camera-stream timestamps within ~0.1 ms.
  This loader deduplicates to keep one timestamp per ≥10 ms window (~30 Hz).

Runs stage 1 of the two-stage pipeline:
  1. python scripts/retarget/hot3d_loader.py --save    -> hot3d_loaded/
  2. python scripts/retarget/hot3d_to_sharpa.py --save -> hot3d_processed/
"""

import argparse
import csv
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mano")

from manotorch.manolayer import ManoLayer  # noqa: E402
from robotic_grounding.retarget import (  # noqa: E402
    HUMAN_MOTION_DATA_DIR,
    MESHES_DIR,
)
from v2d.task_library_loader.lib.dataset_loader_base import (  # noqa: E402
    DatasetLoaderBase,
    SequenceInfo,
    load_meshes_to_device,
    poses_to_root_position_and_axis_angle,
)
from scipy.spatial.transform import Rotation  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="manotorch")

DEFAULT_HOT3D_DIR = HUMAN_MOTION_DATA_DIR / "hot3d" / "dataset"
LOADED_SAVE_DIR = HUMAN_MOTION_DATA_DIR / "hot3d" / "hot3d_loaded"
HOT3D_MESH_DIR = MESHES_DIR / "hot3d"
HOT3D_URDF_DIR = MESHES_DIR.parent / "urdfs" / "hot3d"
HOT3D_FPS = 30.0

# Quest3 world frame is Y-UP; Aria world frame is Z-UP (our pipeline convention).
# Rotate 90° around X to bring Quest3 data into Z-UP:
#   X → X,  Y → Z,  Z → -Y
# Gravity in Quest3: (0, -9.81, 0)  →  after R: (0, 0, -9.81)  ✓
QUEST3_WORLD_TO_ZUP: np.ndarray = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
)

# Minimum nanosecond gap between consecutive frames to keep (10 ms → ~30 Hz).
_MIN_FRAME_GAP_NS = 10_000_000  # 10 ms


def _load_headset_initial_position(
    seq_dir: Path, R_coord: np.ndarray
) -> np.ndarray | None:
    """Read the first frame of headset_trajectory.csv and return the headset position in Z-UP frame.

    Returns None if the file is absent or empty.
    """
    traj_path = seq_dir / "headset_trajectory.csv"
    if not traj_path.exists():
        return None
    with traj_path.open("r") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
    if first_row is None:
        return None
    tx = float(first_row["t_wo_x[m]"])
    ty = float(first_row["t_wo_y[m]"])
    tz = float(first_row["t_wo_z[m]"])
    return (R_coord @ np.array([tx, ty, tz], dtype=np.float32)).astype(np.float32)


def _yaw_cancel_headset_to_scene(
    headset_pos_zup: np.ndarray, scene_center_xy: np.ndarray
) -> np.ndarray:
    """Compute a Z-rotation so the direction from headset to workspace points to +Y.

    The key insight: in each recording session the SLAM/tracking world frame yaw
    is arbitrary.  The headset is at the person's head; the workspace (table) is
    at some horizontal direction from the person.  After normalising, all sessions
    have the workspace in the +Y direction from the headset's starting position,
    giving a consistent orientation for all sequences.

    Args:
        headset_pos_zup:  (3,) headset position in Z-UP world frame.
        scene_center_xy:  (2,) mean XY of all object + hand positions (scene centre
                          in the same Z-UP frame, before scene-offset subtraction).

    Returns:
        3×3 float32 rotation matrix R_yaw such that
        R_yaw @ (scene_center_xy - headset_pos_zup[:2]) ∝ [0, 1].
    """
    delta_xy = scene_center_xy - headset_pos_zup[:2]  # direction toward workspace
    if np.linalg.norm(delta_xy) < 1e-3:
        return np.eye(3, dtype=np.float32)  # degenerate case
    yaw_dir = np.arctan2(delta_xy[1], delta_xy[0])
    # Rotate so that delta_xy points to +Y (yaw = 90°), then add 90° extra.
    cancel_angle = np.pi / 2.0 - yaw_dir + np.pi / 2.0  # = π - yaw_dir
    return Rotation.from_euler("z", cancel_angle).as_matrix().astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset-specific metadata
# ---------------------------------------------------------------------------


@dataclass
class Hot3DSequenceSource:
    """Dataset-specific metadata stored in SequenceInfo.source."""

    seq_dir: Path
    hot3d_dir: Path
    object_uids: list[str]  # from metadata.json "object_uids"
    object_names: list[str]  # from metadata.json "object_names"
    headset: str  # "Aria" or "Quest3" from metadata.json


# ---------------------------------------------------------------------------
# PCA expansion helpers
# ---------------------------------------------------------------------------


def _load_pca_components(
    mano_assets_root: str, ncomps: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Load MANO PCA components for right and left hands using manotorch.

    Returns:
        right_components: (ncomps, 45) PCA basis for right hand.
        left_components:  (ncomps, 45) PCA basis for left hand.
    (The mean pose is NOT returned here — it is handled by the MANO wrapper
    internally when flat_hand_mean=False.)
    """
    right_layer = ManoLayer(
        use_pca=True,
        ncomps=ncomps,
        flat_hand_mean=False,
        side="right",
        mano_assets_root=mano_assets_root,
    )
    left_layer = ManoLayer(
        use_pca=True,
        ncomps=ncomps,
        flat_hand_mean=False,
        side="left",
        mano_assets_root=mano_assets_root,
    )
    right_components = right_layer.th_selected_comps.detach().cpu().numpy()  # (15, 45)
    left_components = left_layer.th_selected_comps.detach().cpu().numpy()  # (15, 45)
    return right_components, left_components


def _expand_pca(pca_coeffs: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Expand PCA coefficients to full 45-DOF finger pose (WITHOUT mean).

    Args:
        pca_coeffs: (N, 15) PCA coefficients.
        components: (15, 45) PCA basis matrix from _load_pca_components.

    Returns:
        (N, 45) full finger pose (deviation from mean). The MANO wrapper with
        flat_hand_mean=False will add the actual mean pose internally.
    """
    return (pca_coeffs @ components).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _deduplicate_timestamps(
    timestamps: list[int], min_gap_ns: int = _MIN_FRAME_GAP_NS
) -> list[int]:
    """Keep at most one timestamp per min_gap_ns window (deduplicates camera streams).

    Hot3D records two camera streams per physical frame (~0.1 ms apart at ~30 Hz).
    This keeps only the first timestamp in each cluster.
    """
    if not timestamps:
        return []
    result = [timestamps[0]]
    for ts in timestamps[1:]:
        if ts - result[-1] >= min_gap_ns:
            result.append(ts)
    return result


def _load_mano_jsonl(jsonl_path: Path) -> dict[int, dict]:
    """Load mano_hand_pose_trajectory.jsonl.

    Returns dict mapping timestamp_ns -> hand_poses dict (keys "0"=left, "1"=right).
    """
    data: dict[int, dict] = {}
    with jsonl_path.open("r") as f:
        for line in f:
            entry = json.loads(line)
            ts = int(entry["timestamp_ns"])
            if ts not in data:
                data[ts] = entry["hand_poses"]
    return data


def _load_object_csv(csv_path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load dynamic_objects.csv.

    Returns dict mapping timestamp_ns -> {object_uid: (4,4) world transform}.
    The transform is T_world_object (world frame pose of the object).
    """
    result: dict[int, dict[str, np.ndarray]] = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp[ns]"])
            uid = row["object_uid"]
            tx = float(row["t_wo_x[m]"])
            ty = float(row["t_wo_y[m]"])
            tz = float(row["t_wo_z[m]"])
            qw = float(row["q_wo_w"])
            qx = float(row["q_wo_x"])
            qy = float(row["q_wo_y"])
            qz = float(row["q_wo_z"])
            R_mat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_mat
            T[:3, 3] = [tx, ty, tz]
            if ts not in result:
                result[ts] = {}
            result[ts][uid] = T
    return result


def _extract_hand_frame(
    hand_entry: dict,
    right_components: np.ndarray,
    left_components: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract (global_orient_rv, finger_pose_45, trans, betas) for both hands.

    hand_entry: {"0": {...left...}, "1": {...right...}} from mano_hand_pose_trajectory.jsonl
    Returns: (rg, rp, rt, rb, lg, lp, lt, lb) — all np.float32.
    """

    def _parse_hand(
        h: dict, components: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pose_pca = np.array(h["pose"], dtype=np.float32)  # (15,)
        w = h["wrist_xform"]
        q_wxyz = np.array(w["q_wxyz"], dtype=np.float32)  # [w,x,y,z]
        t_xyz = np.array(w["t_xyz"], dtype=np.float32)  # (3,)
        betas = np.array(h["betas"], dtype=np.float32)[:10]  # (10,)
        global_orient = (
            Rotation.from_quat(q_wxyz, scalar_first=True).as_rotvec().astype(np.float32)
        )  # (3,)
        finger_pose = _expand_pca(pose_pca[None], components)[0]  # (45,)
        return global_orient, finger_pose, t_xyz, betas

    rg, rp, rt, rb = _parse_hand(hand_entry["1"], right_components)  # "1" = right
    lg, lp, lt, lb = _parse_hand(hand_entry["0"], left_components)  # "0" = left
    return rg, rp, rt, rb, lg, lp, lt, lb


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


class Hot3DDatasetLoader(DatasetLoaderBase):
    """Hot3D dataset loader."""

    def __init__(self) -> None:
        """Initialize caches; MANO layers are built lazily from --mano_model_dir."""
        super().__init__()
        # The PCA MANO layers need the assets dir from --mano_model_dir, which is
        # only available once run() sets self._args. Build them lazily on first
        # use (see _ensure_mano_layers) instead of hardcoding a baked-in path —
        # the OSMO load fetches MANO from swift at runtime.
        self._right_mano_layer: ManoLayer | None = None
        self._left_mano_layer: ManoLayer | None = None
        self._right_components: np.ndarray | None = None
        self._left_components: np.ndarray | None = None
        self._valid_frame_ids_cache: dict[str, list[int]] = {}
        self._scene_offset_cache: dict[str, np.ndarray] = {}
        self._r_total_cache: dict[str, np.ndarray] = (
            {}
        )  # combined coord+yaw rotation per sequence

    def _ensure_mano_layers(self) -> None:
        """Build the PCA MANO layers from --mano_model_dir (once).

        Hot3D stores hand poses as 15-component PCA, so we need ManoLayer both
        for the PCA component matrix (to expand to 45-DOF axis-angle) and for the
        Quest3 J_shaped[0] zero-pose forward. The assets dir comes from
        ``self._args.mano_model_dir`` (set by the base ``run()``), matching the
        args-first contract the rest of the loaders use.
        """
        if self._right_mano_layer is not None:
            return
        mano_assets_root = str(self._args.mano_model_dir)
        self._right_mano_layer = ManoLayer(
            use_pca=True,
            ncomps=15,
            flat_hand_mean=False,
            side="right",
            mano_assets_root=mano_assets_root,
        )
        self._left_mano_layer = ManoLayer(
            use_pca=True,
            ncomps=15,
            flat_hand_mean=False,
            side="left",
            mano_assets_root=mano_assets_root,
        )
        self._right_components = (
            self._right_mano_layer.th_selected_comps.detach().cpu().numpy()
        )
        self._left_components = (
            self._left_mano_layer.th_selected_comps.detach().cpu().numpy()
        )

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """Discover all sequences in hot3d_dir (subdirectories with MANO JSONL)."""
        hot3d_dir = Path(args.dataset_root)
        seq_name_filter = getattr(args, "seq_name", None)

        seq_dirs = sorted(
            p
            for p in hot3d_dir.iterdir()
            if p.is_dir() and (p / "mano_hand_pose_trajectory.jsonl").exists()
        )
        if seq_name_filter:
            seq_dirs = [p for p in seq_dirs if seq_name_filter in p.name]

        out: list[SequenceInfo] = []
        for seq_dir in seq_dirs:
            meta_path = seq_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if not meta.get("have_hand_object_pose_gt", False):
                continue

            object_uids: list[str] = [str(uid) for uid in meta.get("object_uids", [])]
            object_names: list[str] = list(meta.get("object_names", object_uids))
            if not object_uids:
                continue

            headset: str = meta.get("headset", "Aria")  # "Aria" or "Quest3"
            sequence_id = seq_dir.name
            object_name = "+".join(object_names[:3])  # summary name (first 3 objects)
            out.append(
                SequenceInfo(
                    sequence_id=sequence_id,
                    raw_motion_file=sequence_id,
                    object_name=object_name,
                    object_body_names=object_uids,
                    source=Hot3DSequenceSource(
                        seq_dir=seq_dir,
                        hot3d_dir=hot3d_dir,
                        object_uids=object_uids,
                        object_names=object_names,
                        headset=headset,
                    ),
                )
            )
        return out

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO parameters from Hot3D JSONL; expand PCA to 45-DOF."""
        self._ensure_mano_layers()
        src: Hot3DSequenceSource = sequence_info.source
        mano_data = _load_mano_jsonl(src.seq_dir / "mano_hand_pose_trajectory.jsonl")
        obj_data_all = _load_object_csv(src.seq_dir / "dynamic_objects.csv")

        # Deduplicate timestamps (multiple camera streams per real frame)
        all_ts = sorted(mano_data.keys())
        dedup_ts = _deduplicate_timestamps(all_ts)

        # Filter to timestamps where ALL object UIDs have pose data
        valid_ts: list[int] = []
        for ts in dedup_ts:
            if ts not in mano_data:
                continue
            if not all(
                ts in obj_data_all and uid in obj_data_all[ts]
                for uid in src.object_uids
            ):
                continue
            # Both hands must be present
            hp = mano_data[ts]
            if "0" not in hp or "1" not in hp:
                continue
            valid_ts.append(ts)

        if not valid_ts:
            raise ValueError(f"No valid frames in {src.seq_dir.name}")

        # Step 1: Gravity alignment only (Quest3 Y-UP → Z-UP; Aria already Z-UP).
        R_coord = (
            QUEST3_WORLD_TO_ZUP
            if src.headset == "Quest3"
            else np.eye(3, dtype=np.float32)
        )

        # Step 2: Compute scene centre and table z in the gravity-aligned frame.
        # We need the scene centre BEFORE applying yaw so we can use it to determine
        # the per-sequence yaw normalisation (direction from headset to workspace).
        #
        # Z: min object z  →  table surface at z=0 after scene-offset subtraction.
        # XY: mean of all object + wrist positions  →  interaction near xy origin.
        all_positions_Rc: list[np.ndarray] = []
        all_obj_z_Rc: list[float] = []
        for ts in valid_ts:
            for uid in src.object_uids:
                pos = (R_coord @ obj_data_all[ts][uid][:3, 3]).astype(np.float32)
                all_positions_Rc.append(pos)
                all_obj_z_Rc.append(float(pos[2]))
        for ts in valid_ts:
            hp = mano_data[ts]
            for hand_key in ("0", "1"):
                if hand_key in hp:
                    t = R_coord @ np.array(
                        hp[hand_key]["wrist_xform"]["t_xyz"], dtype=np.float32
                    )
                    all_positions_Rc.append(t.astype(np.float32))

        if all_positions_Rc:
            scene_center_xy = np.mean(all_positions_Rc, axis=0)[:2].astype(np.float32)
            z_table = float(np.min(all_obj_z_Rc))
        else:
            scene_center_xy = np.zeros(2, dtype=np.float32)
            z_table = 0.0

        # Step 3: Compute per-sequence yaw normalisation.
        # Strategy: make the direction from the headset's starting position to the
        # workspace centroid point to +Y.  This is more robust than using the
        # headset's "forward" direction because:
        #  - The person looks steeply DOWN at the table, so the horizontal projection
        #    of the forward direction is small and unstable.
        #  - The headset→workspace direction directly encodes where the table is
        #    relative to the person, regardless of headset-specific frame conventions.
        headset_pos_Rc = _load_headset_initial_position(src.seq_dir, R_coord)
        if headset_pos_Rc is not None:
            R_yaw = _yaw_cancel_headset_to_scene(headset_pos_Rc, scene_center_xy)
        else:
            R_yaw = np.eye(3, dtype=np.float32)

        # Step 4: Combined transform (gravity alignment + yaw normalisation).
        R_total = (R_yaw @ R_coord).astype(np.float32)
        self._r_total_cache[sequence_info.sequence_id] = R_total

        # Step 5: Compute scene_offset in R_total frame.
        # Since R_yaw is a pure Z-rotation, z is unchanged: z_table stays the same.
        # Only the xy centroid rotates:  scene_center_xy_Rt = R_yaw[:2,:2] @ scene_center_xy.
        scene_center_xy_Rt = (R_yaw[:2, :2] @ scene_center_xy).astype(np.float32)
        scene_offset = np.array(
            [scene_center_xy_Rt[0], scene_center_xy_Rt[1], z_table], dtype=np.float32
        )

        # Cache for load_object_data
        self._valid_frame_ids_cache[sequence_info.sequence_id] = valid_ts
        self._scene_offset_cache[sequence_info.sequence_id] = scene_offset

        rg_list, rp_list, rt_list = [], [], []
        lg_list, lp_list, lt_list = [], [], []
        right_betas: np.ndarray | None = None
        left_betas: np.ndarray | None = None

        for ts in valid_ts:
            hp = mano_data[ts]
            try:
                rg, rp, rt, rb, lg, lp, lt, lb = _extract_hand_frame(
                    hp, self._right_components, self._left_components
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"Bad frame at ts={ts}: {e}") from e
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

        H = len(valid_ts)
        right_global_orient = np.stack(rg_list).astype(np.float32)
        right_finger_pose = np.stack(rp_list).astype(np.float32)
        right_trans = np.stack(rt_list).astype(np.float32)
        left_global_orient = np.stack(lg_list).astype(np.float32)
        left_finger_pose = np.stack(lp_list).astype(np.float32)
        left_trans = np.stack(lt_list).astype(np.float32)

        # Apply the combined world-frame rotation R_total to all MANO quantities.
        # This covers both Quest3 Y-UP→Z-UP and the per-sequence yaw normalization.
        #
        # global_orient: world-frame change = simple left-multiply R @ R_old.
        #   NOT the similarity transform R @ R_old @ R^T (local-frame change).
        #
        # transl: with center_idx=None, manotorch uses wrist_world = J_shaped[0] + transl.
        #   Rotating the world frame requires:
        #     new_transl = R @ (J_shaped[0] + transl) - J_shaped[0]
        #   Simply doing R @ transl would leave a betas-dependent position error each frame.
        with torch.no_grad():
            rb_t = torch.from_numpy(right_betas).float().unsqueeze(0)
            lb_t = torch.from_numpy(left_betas).float().unsqueeze(0)
            zero_pose = torch.zeros(1, 3 + 15)  # global_orient(3) + PCA comps(15)
            J_right = (
                self._right_mano_layer(pose_coeffs=zero_pose, betas=rb_t)
                .joints[0, 0]
                .numpy()
            )  # (3,) wrist template position
            J_left = (
                self._left_mano_layer(pose_coeffs=zero_pose, betas=lb_t)
                .joints[0, 0]
                .numpy()
            )

        R = R_total
        for i in range(H):
            R_r = Rotation.from_rotvec(right_global_orient[i]).as_matrix()
            right_global_orient[i] = Rotation.from_matrix(R @ R_r).as_rotvec()
            R_l = Rotation.from_rotvec(left_global_orient[i]).as_matrix()
            left_global_orient[i] = Rotation.from_matrix(R @ R_l).as_rotvec()

        right_trans = (R @ (right_trans + J_right).T).T - J_right
        left_trans = (R @ (left_trans + J_left).T).T - J_left

        right_trans -= scene_offset
        left_trans -= scene_offset

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
        """Load object poses aligned to the same valid timestamps as MANO data."""
        src: Hot3DSequenceSource = sequence_info.source
        valid_ts = self._valid_frame_ids_cache.get(sequence_info.sequence_id)
        if valid_ts is None:
            raise RuntimeError(
                "load_mano_data must be called before load_object_data "
                f"(no cached timestamps for '{sequence_info.sequence_id}')"
            )

        obj_data_all = _load_object_csv(src.seq_dir / "dynamic_objects.csv")
        scene_offset = self._scene_offset_cache.get(
            sequence_info.sequence_id, np.zeros(3, dtype=np.float32)
        )

        # Use the same combined rotation (gravity alignment + yaw normalisation) that
        # was applied to the MANO data.  Both rotation and translation use the simple
        # left-multiply R @ old (world-frame change), NOT the similarity transform.
        R_total = self._r_total_cache.get(sequence_info.sequence_id)
        if R_total is None:
            raise RuntimeError(
                "load_mano_data must be called before load_object_data "
                f"(no cached R_total for '{sequence_info.sequence_id}')"
            )

        result: dict[str, Any] = {}
        for uid in src.object_uids:
            poses_raw = np.array(
                [obj_data_all[ts][uid] for ts in valid_ts], dtype=np.float64
            )  # (N, 4, 4)
            R = R_total.astype(np.float64)
            poses_raw[:, :3, :3] = R @ poses_raw[:, :3, :3]
            poses_raw[:, :3, 3] = (R @ poses_raw[:, :3, 3].T).T
            poses_raw[:, :3, 3] -= scene_offset  # apply same centering as hands
            root_pos, root_aa = poses_to_root_position_and_axis_angle(poses_raw)
            result[uid] = (poses_raw, root_pos, root_aa, None)  # None = no articulation

        return result

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple[
        dict[str, Any],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        bool,
    ]:
        """Load Hot3D object meshes from assets/meshes/hot3d/ directory (.glb, meters)."""
        src: Hot3DSequenceSource = sequence_info.source
        assets_dir = Path(self._args.mesh_dir)
        mesh_paths: dict[str, str] = {}
        for uid in src.object_uids:
            glb_path = assets_dir / f"{uid}.glb"
            if glb_path.exists():
                mesh_paths[uid] = str(glb_path)
            else:
                print(f"Warning: No mesh found for object {uid} at {glb_path}")
        return load_meshes_to_device(mesh_paths, device, vertex_scale=1.0)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """Hot3D uses standard MANO (flat_hand_mean=False) without joint centering.

        Hot3D stores t_xyz as the smplx `transl` parameter, where the true wrist
        world position = R_global @ J_template[0](betas) + t_xyz.  Using
        center_idx=None lets manotorch compute this correctly.  With center_idx=0
        the FK_joint0 term is dropped, causing the hand to float away from objects.
        """
        return {"flat_hand_mean": False, "center_idx": None}

    def get_fps(self) -> float:
        """Return Hot3D effective frame rate after deduplication."""
        return HOT3D_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return GLB mesh paths for all objects in the sequence."""
        src: Hot3DSequenceSource = sequence_info.source
        assets_dir = Path(self._args.mesh_dir)
        return [str(assets_dir / f"{uid}.glb") for uid in src.object_uids]

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return rigid URDF paths for all objects in the sequence.

        URDFs are generated by scripts/generate_rigid_urdfs.py --dataset hot3d.
        """
        src: Hot3DSequenceSource = sequence_info.source
        urdf_dir = Path(self._args.object_model_root)
        return [str(urdf_dir / f"{uid}_rigid.urdf") for uid in src.object_uids]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Hot3D loader script."""
    parser = argparse.ArgumentParser(
        description="Load Hot3D sequences into ManoSharpaData schema (MANO + object only)."
    )
    DatasetLoaderBase.add_common_args(
        parser,
        dataset_root=DEFAULT_HOT3D_DIR,
        object_model_root=HOT3D_URDF_DIR,
        mesh_dir=HOT3D_MESH_DIR,
        output_dir=LOADED_SAVE_DIR,
    )
    # Hot3D-specific extras.
    parser.add_argument("--mano_to_robot_scale", type=float, default=1.2)
    parser.add_argument(
        "--seq_name",
        type=str,
        default=None,
        help="Process only sequences whose folder name contains this substring.",
    )
    parser.add_argument(
        "--list_sequences",
        action="store_true",
        default=False,
        help="List available sequences and exit.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the Hot3D loader: list sequences or process and save ManoSharpaData Parquet files."""
    if args.list_sequences:
        loader = Hot3DDatasetLoader()
        sequences = loader.list_sequences(args)
        print(f"Found {len(sequences)} sequences in {args.dataset_root}")
        for i, s in enumerate(sequences, start=1):
            print(f"{i:3d}. {s.sequence_id}  objects={s.object_name}")
        return

    loader = Hot3DDatasetLoader()
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
