# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load DexYCB dataset (NVLabs, CVPR 2021) into ManoSharpaData schema.

DexYCB layout (after extraction, ``$DEX_YCB_DIR``):

  dexycb_dir/
    20200709-subject-01/
      20200709_141841/                    # session id
        meta.yml                          # ycb_ids, ycb_grasp_ind, mano_side, mano_calib
        {camera_serial}/                   # 8 cameras per session
          color_{N:06d}.jpg                (ignored here)
          aligned_depth_to_color_{N:06d}.png (ignored)
          labels_{N:06d}.npz               # per-frame pose_y / pose_m / joint_3d / seg
    calibration/
      mano_{calib_id}/mano.yml             # per-subject MANO betas
      intrinsics/{camera_serial}_*.yml
      extrinsics_{ext_id}/extrinsics.yml
    models/
      {ycb_name}/textured_simple.obj       # 21 YCB object meshes

Per-frame ``labels_*.npz`` keys we consume:
  pose_y  : (num_obj, 3, 4)  — ``[R | t]`` for each YCB object in CAMERA frame.
  pose_m  : (1, 51)          — 3 global_orient + 45 PCA finger pose + 3 trans.

MANO format:
  ``pose_m[:, 0:48]`` is MANO in PCA representation (ncomps=45, flat_hand_mean=False).
  We expand the 45 PCA coefs to 45-DOF axis-angle before handing to MANO FK,
  matching the pattern in ``hot3d_loader.py``.

Single-hand sequences:
  Each DexYCB session uses exactly one hand (``meta['mano_side']`` in
  {right, left}). The other hand is filled with a zero pose at the origin
  so the retargeter can still produce a two-hand robot trajectory (the
  idle arm just sits at the world origin).

Coordinate frame:
  Poses are in the chosen camera's frame. We rotate to a Z-up world via
  the same ``CAM_TO_WORLD_ZUP`` used for H2O. This is an approximation —
  switch cameras or refine the extrinsics if the visualisation shows the
  scene tilted.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import tarfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mano")

from manotorch.manolayer import ManoLayer  # noqa: E402
from robotic_grounding.retarget import (  # noqa: E402
    ASSETS_DIR,
    BODY_MODELS_DIR,
    HUMAN_MOTION_DATA_DIR,
    MESHES_DIR,
)
from robotic_grounding.retarget.dataset_loader_base import (  # noqa: E402
    DatasetLoaderBase,
    SequenceInfo,
    load_meshes_to_device,
    make_usd_safe,
)

logging.getLogger().setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DEXYCB_DIR = HUMAN_MOTION_DATA_DIR / "dexycb" / "dataset"
LOADED_SAVE_DIR = HUMAN_MOTION_DATA_DIR / "dexycb" / "dexycb_loaded"
DEXYCB_FPS = 30.0

# Camera serials (from dex-ycb-toolkit _SERIALS).
DEXYCB_SERIALS: tuple[str, ...] = (
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122062010",
    "932122062940",
)
# Default to the last serial (master in the toolkit).
DEFAULT_CAMERA_SERIAL = "932122062010"

# YCB object ID -> folder name (matches ``dex_ycb_toolkit.dex_ycb._YCB_CLASSES``).
DEXYCB_OBJECTS: dict[int, str] = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

# Camera frame (+X right, +Y down, +Z forward per OpenCV) -> Z-up world.
# We follow the same form as ``h2o_loader.CAM_TO_WORLD_ZUP``; revisit if
# DexYCB sequences come out tilted.
CAM_TO_WORLD_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DexYCBSequenceSource:
    """Source info for one DexYCB sequence (subject/session/camera triple)."""

    session_dir: Path  # .../{subject}/{session}/
    camera_dir: Path  # .../{subject}/{session}/{serial}/
    subject: str  # e.g. "20200709-subject-01"
    session: str  # e.g. "20200709_141841"
    camera_serial: str
    ycb_ids: list[int]
    ycb_grasp_ind: int
    ycb_names: list[str]  # derived from DEXYCB_OBJECTS
    mano_side: str  # "right" | "left"
    mano_calib: str


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def _read_meta(session_dir: Path) -> dict[str, Any]:
    with open(session_dir / "meta.yml", "r") as f:
        return yaml.safe_load(f)


def _read_mano_betas(dexycb_dir: Path, mano_calib: str) -> np.ndarray:
    """Load per-subject MANO betas from calibration/mano_{id}/mano.yml."""
    path = dexycb_dir / "calibration" / f"mano_{mano_calib}" / "mano.yml"
    with open(path, "r") as f:
        calib = yaml.safe_load(f)
    return np.asarray(calib["betas"], dtype=np.float32)


def _collect_frame_ids(camera_dir: Path) -> list[str]:
    """Return sorted frame ids (stems like '000042') present in camera_dir."""
    return sorted(p.stem.split("_", 1)[1] for p in camera_dir.glob("labels_*.npz"))


def _expand_pca_to_aa(
    pca_coeffs: np.ndarray, components: np.ndarray, hands_mean: np.ndarray
) -> np.ndarray:
    """Expand (N, 45) PCA coefficients to (N, 45) axis-angle finger pose.

    ``hands_mean`` is added because DexYCB was fitted with
    ``flat_hand_mean=False`` so the mean pose is baked into the reconstruction.
    """
    return pca_coeffs @ components + hands_mean[None, :]


def _extract_archives_if_needed(dexycb_dir: Path) -> Path:
    """Extract ``{name}.tar.gz`` bundles in ``dexycb_dir`` if not yet expanded.

    The CSS upload ships per-subject tarballs (plus ``calibration.tar.gz``
    and ``models.tar.gz``) instead of the 600k+ raw label/mesh files, to
    keep S3 round-trips down.  Each tarball was created with entries rooted
    at its namesake directory (e.g. ``20200709-subject-01/...``) so
    extracting with ``-C dexycb_dir`` restores the canonical layout.

    Idempotent: skips any tarball whose extraction target directory already
    exists.  If the input dir is read-only (e.g. the CSS bind mount), the
    extraction is redirected to a ``/tmp`` cache.
    """
    archives = sorted(dexycb_dir.glob("*.tar.gz"))
    if not archives:
        return dexycb_dir

    def _already_extracted(archive: Path) -> bool:
        stem = archive.name[: -len(".tar.gz")]
        return (dexycb_dir / stem).is_dir()

    pending = [a for a in archives if not _already_extracted(a)]
    if not pending:
        return dexycb_dir

    # Serialize extraction across shards via an advisory file lock — the
    # first shard extracts while the others block, then re-check and find
    # `_already_extracted` True.
    lock_path = Path("/tmp") / f"dexycb_extract_{dexycb_dir.name}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        pending = [a for a in archives if not _already_extracted(a)]
        if not pending:
            return dexycb_dir

        target = dexycb_dir
        try:
            probe = dexycb_dir / ".extract_test"
            probe.touch()
            probe.unlink()
        except OSError:
            target = Path("/tmp") / f"dexycb_extracted_{dexycb_dir.name}"
            target.mkdir(parents=True, exist_ok=True)
            print(f"[dexycb] input dir read-only, extracting to cache: {target}")

        for archive in pending:
            print(f"[dexycb] extracting {archive.name} -> {target}")
            with tarfile.open(archive, "r:gz") as t:
                t.extractall(target)

        return target


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class DexYCBDatasetLoader(DatasetLoaderBase):
    """Load DexYCB sequences into ManoSharpaData (MANO + object only)."""

    def __init__(self) -> None:
        """Pre-build right/left ManoLayers so FK is ready before list_sequences."""
        self._args: argparse.Namespace | None = None
        mano_assets_root = str(BODY_MODELS_DIR / "mano")
        self._right_layer = ManoLayer(
            use_pca=True,
            ncomps=45,
            flat_hand_mean=False,
            side="right",
            mano_assets_root=mano_assets_root,
        )
        self._left_layer = ManoLayer(
            use_pca=True,
            ncomps=45,
            flat_hand_mean=False,
            side="left",
            mano_assets_root=mano_assets_root,
        )
        self._right_components = (
            self._right_layer.th_selected_comps.detach().cpu().numpy()
        )
        self._left_components = (
            self._left_layer.th_selected_comps.detach().cpu().numpy()
        )
        self._right_hands_mean = (
            self._right_layer.th_hands_mean.detach().cpu().numpy().squeeze()
        )
        self._left_hands_mean = (
            self._left_layer.th_hands_mean.detach().cpu().numpy().squeeze()
        )

    # ------------------------------------------------------------------
    # Sequence discovery
    # ------------------------------------------------------------------
    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """Discover DexYCB sequences (one per session for the chosen camera)."""
        self._args = args
        dexycb_dir = Path(getattr(args, "dexycb_dir", DEFAULT_DEXYCB_DIR))
        serial = getattr(args, "camera_serial", DEFAULT_CAMERA_SERIAL)

        if not dexycb_dir.exists():
            raise FileNotFoundError(f"DexYCB dataset not found at {dexycb_dir}")

        # If the dataset ships as per-subject tarballs (e.g. on CSS), expand
        # them once before walking the tree.
        dexycb_dir = _extract_archives_if_needed(dexycb_dir)

        sequences: list[SequenceInfo] = []
        for subject_dir in sorted(dexycb_dir.iterdir()):
            if not subject_dir.is_dir() or "-subject-" not in subject_dir.name:
                continue
            subject = subject_dir.name
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                camera_dir = session_dir / serial
                if not camera_dir.exists():
                    continue
                try:
                    meta = _read_meta(session_dir)
                except FileNotFoundError:
                    continue

                ycb_ids = list(meta.get("ycb_ids", []))
                ycb_names = [
                    DEXYCB_OBJECTS.get(cid, f"object_{cid}") for cid in ycb_ids
                ]
                if not ycb_ids:
                    continue
                sequence_id = f"dexycb_{subject}_{session_dir.name}_{serial}"

                source = DexYCBSequenceSource(
                    session_dir=session_dir,
                    camera_dir=camera_dir,
                    subject=subject,
                    session=session_dir.name,
                    camera_serial=serial,
                    ycb_ids=ycb_ids,
                    ycb_grasp_ind=int(meta.get("ycb_grasp_ind", 0)),
                    ycb_names=ycb_names,
                    mano_side=str(meta.get("mano_side", "right")),
                    mano_calib=(
                        str(meta.get("mano_calib", [""])[0])
                        if isinstance(meta.get("mano_calib"), list)
                        else str(meta.get("mano_calib", ""))
                    ),
                )
                sequences.append(
                    SequenceInfo(
                        sequence_id=sequence_id,
                        raw_motion_file=str(camera_dir),
                        object_name="+".join(ycb_names),
                        object_body_names=ycb_names,
                        source=source,
                    )
                )

        sequences = self._apply_sequence_filters(sequences, args)
        print(f"Found {len(sequences)} DexYCB sequences")
        return sequences

    # ------------------------------------------------------------------
    # MANO
    # ------------------------------------------------------------------
    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Parse pose_m from every labels_*.npz frame in the sequence."""
        src: DexYCBSequenceSource = sequence_info.source
        dexycb_dir = Path(getattr(self._args, "dexycb_dir", DEFAULT_DEXYCB_DIR))
        frame_ids = _collect_frame_ids(src.camera_dir)
        if not frame_ids:
            raise FileNotFoundError(f"No labels_*.npz in {src.camera_dir}")

        active_side = src.mano_side
        assert active_side in {"right", "left"}, f"Bad mano_side {active_side}"

        g_list: list[np.ndarray] = []  # (3,)
        p_list: list[np.ndarray] = []  # (45,) PCA
        t_list: list[np.ndarray] = []  # (3,)
        for fid in frame_ids:
            arr = np.load(src.camera_dir / f"labels_{fid}.npz")
            pose_m = arr["pose_m"]  # (1, 51)
            if pose_m.shape != (1, 51):
                raise ValueError(
                    f"{src.camera_dir}/labels_{fid}.npz: pose_m shape {pose_m.shape}"
                )
            g_list.append(pose_m[0, 0:3].astype(np.float32))
            p_list.append(pose_m[0, 3:48].astype(np.float32))
            t_list.append(pose_m[0, 48:51].astype(np.float32))

        H = len(frame_ids)
        global_orient = np.stack(g_list)  # (H, 3)
        pca_pose = np.stack(p_list)  # (H, 45)
        trans = np.stack(t_list)  # (H, 3)

        # Expand PCA -> axis-angle in the active hand's basis.
        if active_side == "right":
            finger_pose = _expand_pca_to_aa(
                pca_pose, self._right_components, self._right_hands_mean
            )
        else:
            finger_pose = _expand_pca_to_aa(
                pca_pose, self._left_components, self._left_hands_mean
            )
        finger_pose = finger_pose.astype(np.float32)

        # Camera frame -> Z-up world.
        R = CAM_TO_WORLD_ZUP
        for i in range(H):
            Rm = Rotation.from_rotvec(global_orient[i]).as_matrix()
            global_orient[i] = Rotation.from_matrix(R @ Rm).as_rotvec()
        trans = (R @ trans.T).T

        betas = _read_mano_betas(dexycb_dir, src.mano_calib)

        # Fill the idle hand with zeros (pose at origin, flat default shape).
        zero_g = np.zeros((H, 3), dtype=np.float32)
        zero_p = np.zeros((H, 45), dtype=np.float32)
        zero_t = np.zeros((H, 3), dtype=np.float32)
        zero_b = np.zeros(10, dtype=np.float32)

        active = {
            "global_orient": global_orient.astype(np.float32),
            "finger_pose": finger_pose,
            "trans": trans.astype(np.float32),
            "betas": betas.astype(np.float32),
        }
        idle = {
            "global_orient": zero_g,
            "finger_pose": zero_p,
            "trans": zero_t,
            "betas": zero_b,
        }
        right = active if active_side == "right" else idle
        left = active if active_side == "left" else idle

        return {
            "H": H,
            "right_global_orient": torch.from_numpy(right["global_orient"]).to(device),
            "right_finger_pose": torch.from_numpy(right["finger_pose"]).to(device),
            "right_trans": torch.from_numpy(right["trans"]).to(device),
            "right_betas": torch.from_numpy(right["betas"]).to(device),
            "right_fitting_err": torch.zeros(H, device=device),
            "left_global_orient": torch.from_numpy(left["global_orient"]).to(device),
            "left_finger_pose": torch.from_numpy(left["finger_pose"]).to(device),
            "left_trans": torch.from_numpy(left["trans"]).to(device),
            "left_betas": torch.from_numpy(left["betas"]).to(device),
            "left_fitting_err": torch.zeros(H, device=device),
        }

    # ------------------------------------------------------------------
    # Object poses
    # ------------------------------------------------------------------
    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Parse pose_y 3x4 matrices across all frames, rotate to Z-up."""
        src: DexYCBSequenceSource = sequence_info.source
        frame_ids = _collect_frame_ids(src.camera_dir)
        N = len(frame_ids)
        M = len(src.ycb_ids)

        poses = np.tile(np.eye(4, dtype=np.float32), (M, N, 1, 1))  # (M, N, 4, 4)
        last = [np.eye(4, dtype=np.float32) for _ in range(M)]
        R = CAM_TO_WORLD_ZUP

        for f_idx, fid in enumerate(frame_ids):
            arr = np.load(src.camera_dir / f"labels_{fid}.npz")
            pose_y = arr["pose_y"]  # (num_obj_in_file, 3, 4)
            if pose_y.shape[0] != M:
                # Shouldn't happen if meta.ycb_ids is consistent, but be safe.
                pose_y = pose_y[:M]
            for k in range(M):
                Rcam = pose_y[k, :3, :3]
                tcam = pose_y[k, :3, 3]
                if np.allclose(Rcam, 0) and np.allclose(tcam, 0):
                    # Missing / un-labelled frame for this object: hold last.
                    poses[k, f_idx] = last[k]
                    continue
                Twc = np.eye(4, dtype=np.float32)
                Twc[:3, :3] = R @ Rcam
                Twc[:3, 3] = R @ tcam
                poses[k, f_idx] = Twc
                last[k] = Twc

        result: dict[str, Any] = {}
        for k, name in enumerate(src.ycb_names):
            pose_seq = poses[k]
            root_pos = pose_seq[:, :3, 3].copy()
            root_aa = np.stack(
                [Rotation.from_matrix(T[:3, :3]).as_rotvec() for T in pose_seq]
            ).astype(np.float32)
            result[name] = (pose_seq, root_pos, root_aa, None)
        return result

    # ------------------------------------------------------------------
    # Meshes
    # ------------------------------------------------------------------
    def load_object_meshes(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> tuple:
        """Load the YCB textured meshes for each object in the scene.

        Searches in order:
        1. ``assets/meshes/dexycb/{name}/textured_simple.obj`` (committed copy)
        2. ``{dexycb_dir}/models/{name}/textured_simple.obj`` (canonical)
        """
        src: DexYCBSequenceSource = sequence_info.source
        dexycb_dir = Path(getattr(self._args, "dexycb_dir", DEFAULT_DEXYCB_DIR))
        canonical_dir = MESHES_DIR / "dexycb"
        models_dir = dexycb_dir / "models"

        mesh_paths: dict[str, str] = {}
        missing: list[str] = []
        for name in src.ycb_names:
            chosen: Path | None = None
            for base in (canonical_dir, models_dir):
                preferred = base / name / "textured_simple.obj"
                if preferred.exists():
                    chosen = preferred
                    break
                objs = (
                    sorted((base / name).glob("*.obj"))
                    if (base / name).is_dir()
                    else []
                )
                if objs:
                    chosen = objs[0]
                    break
            if chosen is None:
                missing.append(name)
                continue
            mesh_paths[name] = str(chosen)

        if missing:
            raise FileNotFoundError(
                f"DexYCB meshes missing for {sequence_info.sequence_id}: "
                f"{missing}. Searched {canonical_dir} and {models_dir}."
            )
        return load_meshes_to_device(mesh_paths, device, vertex_scale=1.0)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """We pre-expand PCA -> axis-angle, so MANO FK runs with use_pca=False."""
        return {"flat_hand_mean": False, "center_idx": None}

    def get_fps(self) -> float:
        """Return DexYCB capture rate (30 Hz)."""
        return DEXYCB_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return per-object mesh paths for a sequence, preferring committed copies."""
        src: DexYCBSequenceSource = sequence_info.source
        mesh_dir = MESHES_DIR / "dexycb"
        paths: list[str] = []
        for name in src.ycb_names:
            preferred = mesh_dir / name / "textured_simple.obj"
            if preferred.exists():
                paths.append(str(preferred))
                continue
            dexycb_dir = Path(getattr(self._args, "dexycb_dir", DEFAULT_DEXYCB_DIR))
            fallback = dexycb_dir / "models" / name / "textured_simple.obj"
            paths.append(str(fallback))
        return paths

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return per-object URDF paths under ``assets/urdfs/dexycb/``."""
        src: DexYCBSequenceSource = sequence_info.source
        urdf_dir = ASSETS_DIR / "urdfs" / "dexycb"
        return [
            str(urdf_dir / f"{make_usd_safe(name)}_rigid.urdf")
            for name in src.ycb_names
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI args for the DexYCB loader."""
    parser = argparse.ArgumentParser(
        description="Load DexYCB sequences into ManoSharpaData schema."
    )
    parser.add_argument(
        "--dexycb_dir",
        type=Path,
        default=DEFAULT_DEXYCB_DIR,
        help="Root directory of the extracted DexYCB dataset (parent of subjects + calibration + models).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=LOADED_SAVE_DIR,
    )
    parser.add_argument(
        "--camera_serial",
        type=str,
        default=DEFAULT_CAMERA_SERIAL,
        help="Which of the 8 DexYCB camera serials to ingest (default: toolkit master).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument(
        "--list_sequences",
        action="store_true",
        help="List sequences and exit.",
    )
    DatasetLoaderBase.add_filter_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the DexYCB loader or list sequences per CLI args."""
    loader = DexYCBDatasetLoader()
    if args.list_sequences:
        for s in loader.list_sequences(args):
            print(s.sequence_id)
        return
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
