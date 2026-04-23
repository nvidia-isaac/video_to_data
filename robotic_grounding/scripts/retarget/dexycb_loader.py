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
  ``pose_y`` / ``pose_m`` live in the chosen camera's frame. DexYCB's
  per-session ``calibration/extrinsics_{ext_id}/extrinsics.yml`` stores a
  3x4 world-to-camera matrix per serial; "world" there is the master
  camera's frame. The master is the overhead camera in DexYCB's rig, so
  its +Y (image-down) is approximately gravity. We compose
  ``R_master_to_zup @ inv(W2C_serial)`` per sequence: first undo the
  serial's extrinsic to land in master frame, then rotate so +Y_master
  (down) -> -Z_world and +Z_master (forward) -> +Y_world. This is not
  exact if the master isn't perfectly vertical, but eliminates the
  fixed-rotation approximation that caused "through the floor" artifacts
  for sequences whose master camera differed from the first-fitted one.
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

# Fallback fixed rotation from the master camera's OpenCV frame (+X right,
# +Y down, +Z forward) to a gravity-aligned Z-up world.  Used only if the
# data-driven gravity estimate below fails (e.g. a labels_*.npz with no
# valid object poses).  Empirically off by ~49° for real DexYCB sessions,
# which is what the data-driven path fixes — keep it as a last-resort
# default so the loader doesn't crash mid-stream.
_MASTER_TO_ZUP_FALLBACK = np.array(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    dtype=np.float32,
)

# Per-extrinsics-id cache for the data-driven master→Z-up rotation.  All
# sessions sharing the same ``extrinsics_<id>`` calibration file share the
# same rig geometry and therefore the same gravity direction in master,
# so we memoize on the id and re-use across sequences.
_GRAVITY_ALIGN_CACHE: dict[str, np.ndarray] = {}


def _rotation_align_to_zup(gravity_up_in_master: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation R such that ``R @ gravity_up_in_master == +Z``.

    Leaves the perpendicular-to-gravity directions well-defined (Rodrigues
    rotation around the cross product), so repeated calls on the same
    gravity vector give the same result.
    """
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = np.asarray(gravity_up_in_master, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-12)
    axis = np.cross(up, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        # Already aligned (or antipodal — flip via 180° around +X).
        if np.dot(up, target) > 0:
            return np.eye(3, dtype=np.float32)
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    angle = np.arccos(np.clip(np.dot(up, target), -1.0, 1.0))
    return Rotation.from_rotvec(axis / axis_norm * angle).as_matrix().astype(np.float32)


def _estimate_gravity_up_in_master(
    dexycb_dir: Path,
    extrinsics_id: str,
    camera_serial: str,
    camera_dir: Path,
    max_frames: int = 20,
) -> np.ndarray:
    """Derive gravity-up (in master frame) from object rest poses.

    YCB's ``textured_simple.obj`` meshes all share the same canonical
    frame (+Z = up, verified empirically — four different-shape objects in
    one test sequence had local +Z within 2° of each other).  When an
    object sits on a horizontal surface, its local +Z in world frame is
    exactly gravity-up.  We read up to ``max_frames`` ``labels_*.npz``
    files from ``camera_dir``, pull every non-zero ``pose_y[k, :3, 2]``
    (object-local +Z in cam frame), transform to master via the
    pre-computed ``cam_to_master``, and average.  Any motion during the
    sequence averages out; the mean vector is robust across typical
    DexYCB sessions.

    Falls back to the ``_MASTER_TO_ZUP_FALLBACK`` convention (OpenCV Y-down
    = gravity) if no valid pose is found — see the docstring comment.
    """
    ext_path = (
        dexycb_dir / "calibration" / f"extrinsics_{extrinsics_id}" / "extrinsics.yml"
    )
    with open(ext_path, "r") as f:
        calib = yaml.load(f, Loader=yaml.Loader)
    flat = np.asarray(calib["extrinsics"][camera_serial], dtype=np.float64)
    world_to_cam = np.eye(4, dtype=np.float64)
    world_to_cam[:3, :] = flat.reshape(3, 4)
    R_cam_to_master = np.linalg.inv(world_to_cam)[:3, :3]

    ups_in_master: list[np.ndarray] = []
    for fpath in sorted(camera_dir.glob("labels_*.npz"))[:max_frames]:
        try:
            arr = np.load(fpath)
            pose_y = arr["pose_y"]  # (num_obj, 3, 4)
        except Exception:  # noqa: BLE001
            continue
        for k in range(pose_y.shape[0]):
            R_cam = pose_y[k, :3, :3]
            t_cam = pose_y[k, :3, 3]
            if np.allclose(R_cam, 0) and np.allclose(t_cam, 0):
                continue  # un-labelled object in this frame
            obj_up_cam = R_cam[:, 2].astype(np.float64)  # object local +Z
            ups_in_master.append(R_cam_to_master @ obj_up_cam)

    if not ups_in_master:
        warnings.warn(
            f"No valid pose_y rotations under {camera_dir} for extrinsics "
            f"{extrinsics_id!r}; falling back to the fixed master->Zup "
            "approximation (likely ~45° off gravity).",
            stacklevel=2,
        )
        # The fallback matrix was built assuming master +Y is gravity;
        # gravity-up in master is therefore master -Y.
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)

    mean_up = np.mean(np.stack(ups_in_master, axis=0), axis=0)
    return mean_up / np.linalg.norm(mean_up)


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
    extrinsics_id: str  # e.g. "20200702_151821"; indexes calibration/extrinsics_<id>/


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


def _load_cam_to_world(
    dexycb_dir: Path,
    extrinsics_id: str,
    camera_serial: str,
    camera_dir: Path | None = None,
) -> np.ndarray:
    """Return the camera-to-world 4x4 transform for one session.

    DexYCB's ``calibration/extrinsics_<id>/extrinsics.yml`` stores, per
    serial, a 3x4 ``[R | t]`` world-to-camera matrix where "world" is the
    master camera's frame.  We invert to get camera-to-master, then apply
    a per-``extrinsics_id`` rotation that takes master frame into Isaac
    Sim's Z-up world.

    The master→Z-up rotation is **data-driven** per calibration batch:
    :func:`_estimate_gravity_up_in_master` infers gravity from object
    rest poses (YCB canonical +Z = gravity-up), and
    :func:`_rotation_align_to_zup` turns that direction into a 3×3
    rotation matrix.  Result is cached in ``_GRAVITY_ALIGN_CACHE`` so the
    estimate only runs once per batch.

    ``camera_dir`` is required to compute the gravity estimate; it's
    optional only for legacy callers that already have a warmed cache.
    """
    path = dexycb_dir / "calibration" / f"extrinsics_{extrinsics_id}" / "extrinsics.yml"
    with open(path, "r") as f:
        calib = yaml.load(f, Loader=yaml.Loader)  # tagged python/tuple
    extrinsics = calib["extrinsics"]
    if camera_serial not in extrinsics:
        raise KeyError(
            f"Camera serial {camera_serial!r} not found in {path}. "
            f"Available: {sorted(extrinsics)}"
        )
    flat = np.asarray(extrinsics[camera_serial], dtype=np.float64)
    if flat.size != 12:
        raise ValueError(f"{path}: expected 12 floats per serial, got {flat.size}")
    world_to_cam = np.eye(4, dtype=np.float64)
    world_to_cam[:3, :] = flat.reshape(3, 4)
    cam_to_master = np.linalg.inv(world_to_cam)

    if extrinsics_id not in _GRAVITY_ALIGN_CACHE:
        if camera_dir is None:
            # No data available to estimate gravity — fall back to the
            # fixed rotation.  Should only happen if a legacy caller
            # invokes this without camera_dir on a cold cache.
            warnings.warn(
                f"_load_cam_to_world called for extrinsics {extrinsics_id!r} "
                "without camera_dir; using fixed master->Zup fallback.",
                stacklevel=2,
            )
            _GRAVITY_ALIGN_CACHE[extrinsics_id] = _MASTER_TO_ZUP_FALLBACK
        else:
            gravity_up = _estimate_gravity_up_in_master(
                dexycb_dir, extrinsics_id, camera_serial, camera_dir
            )
            _GRAVITY_ALIGN_CACHE[extrinsics_id] = _rotation_align_to_zup(gravity_up)

    master_to_zup = np.eye(4, dtype=np.float64)
    master_to_zup[:3, :3] = _GRAVITY_ALIGN_CACHE[extrinsics_id].astype(np.float64)
    return (master_to_zup @ cam_to_master).astype(np.float32)


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
        # Per-sequence camera-to-world extrinsic cache, shared between
        # load_mano_data and load_object_data so the yml is read once.
        self._cam_to_world_cache: dict[str, np.ndarray] = {}
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

                mano_sides = meta.get("mano_sides", meta.get("mano_side", "right"))
                if isinstance(mano_sides, list):
                    mano_side = str(mano_sides[0]) if mano_sides else "right"
                else:
                    mano_side = str(mano_sides)
                source = DexYCBSequenceSource(
                    session_dir=session_dir,
                    camera_dir=camera_dir,
                    subject=subject,
                    session=session_dir.name,
                    camera_serial=serial,
                    ycb_ids=ycb_ids,
                    ycb_grasp_ind=int(meta.get("ycb_grasp_ind", 0)),
                    ycb_names=ycb_names,
                    mano_side=mano_side,
                    mano_calib=(
                        str(meta.get("mano_calib", [""])[0])
                        if isinstance(meta.get("mano_calib"), list)
                        else str(meta.get("mano_calib", ""))
                    ),
                    extrinsics_id=str(meta.get("extrinsics", "")),
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
        betas = _read_mano_betas(dexycb_dir, src.mano_calib)
        active_layer = self._right_layer if active_side == "right" else self._left_layer
        # manotorch (center_idx=None) uses wrist_world = J0 + transl, so when we
        # rotate the world frame we must rotate that wrist anchor too.  If we
        # only do R @ transl, the hand drifts away from the objects after the
        # gravity-alignment rotation.
        with torch.no_grad():
            zero_pose = torch.zeros(1, 3 + 45, dtype=torch.float32)
            active_joint0 = (
                active_layer(
                    pose_coeffs=zero_pose,
                    betas=torch.from_numpy(betas).float().unsqueeze(0),
                )
                .joints[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        # Camera frame -> gravity-aligned Z-up world via per-session
        # extrinsic composed with master-cam->Z-up rotation. We also fold a
        # per-sequence Z lift into c2w so the lowest point in the scene
        # (object or active wrist) lands just above Isaac Sim's floor
        # plane at z = 0; without it, sequences where the object rests at
        # tabletop below the master-frame origin clip into the ground and
        # drive continuous physics oscillation (aka hand shake on grasp).
        c2w = _load_cam_to_world(
            dexycb_dir, src.extrinsics_id, src.camera_serial, src.camera_dir
        )
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        # Provisional active-hand trans in world frame.  The J0 correction is
        # required for consistency with the rotated MANO global_orient.
        trans_world = (R @ (trans + active_joint0).T).T + t - active_joint0
        # Peek at every frame's object origins (pose_y[k, :3, 3]) and
        # transform them to world; combined with the hand trans this gives
        # the scene's world-frame Z extent.
        obj_z_world: list[float] = []
        for fid in frame_ids:
            arr = np.load(src.camera_dir / f"labels_{fid}.npz")
            pose_y = arr["pose_y"]  # (num_obj, 3, 4)
            for k in range(pose_y.shape[0]):
                tcam = pose_y[k, :3, 3]
                Rcam = pose_y[k, :3, :3]
                if np.allclose(tcam, 0) and np.allclose(Rcam, 0):
                    continue
                obj_z_world.append(float((R @ tcam + t)[2]))
        scene_min_z = min(
            (*obj_z_world, float(trans_world[:, 2].min()))
            if obj_z_world
            else (float(trans_world[:, 2].min()),)
        )
        # Match h2o_loader's generous margin so the scene floats comfortably
        # above Isaac Sim's floor regardless of which mesh half-extent
        # happens to straddle the reconstruct ground-filter (0.05 m).  YCB
        # objects are smaller than H2O (tallest ~20 cm), but the same
        # "centroid vs vertex bottom" mismatch still applies for thin/tall
        # items (e.g. 019_pitcher_base, 011_banana) — 1 m is empirically
        # safe across both datasets and dummy_agent's auto-framed camera
        # hides the absolute offset anyway.
        _GROUND_MARGIN = 1.0  # m above Isaac Sim's floor plane
        z_lift = max(0.0, _GROUND_MARGIN - scene_min_z)
        c2w[2, 3] += z_lift
        t = c2w[:3, 3]
        self._cam_to_world_cache[sequence_info.sequence_id] = c2w
        for i in range(H):
            Rm = Rotation.from_rotvec(global_orient[i]).as_matrix()
            global_orient[i] = Rotation.from_matrix(R @ Rm).as_rotvec()
        trans = (R @ (trans + active_joint0).T).T + t - active_joint0

        # Fill the idle hand with a zero pose parked below the ground plane.
        # DexYCB is single-handed (``mano_side`` selects which); if the idle
        # hand's MANO trans is zero, downstream IK lands its robot at the
        # zero-pose J0 offset (~the world origin), which spawns a second
        # PD-controlled Sharpa in full camera view whose tiny tracking-error
        # oscillation looks like shaking. Parking at z = -0.5 m hides the
        # idle robot under Isaac Sim's floor plane without dragging the
        # auto-framed viewer off the active hand's work area.
        zero_g = np.zeros((H, 3), dtype=np.float32)
        zero_p = np.zeros((H, 45), dtype=np.float32)
        zero_t = np.tile(np.array([0.0, 0.0, -0.5], dtype=np.float32), (H, 1))
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
        c2w = self._cam_to_world_cache.get(sequence_info.sequence_id)
        if c2w is None:
            # Cold-cache fallback.  Reloads extrinsic + re-estimates gravity
            # via the per-batch cache inside ``_load_cam_to_world``, so the
            # result includes the correct master->Z-up rotation — but NOT
            # the ``z_lift`` applied in ``load_mano_data``, which depends on
            # hand trans and is only computed when MANO data is loaded
            # first.  Intended as a safety net, not the common path.
            dexycb_dir = Path(getattr(self._args, "dexycb_dir", DEFAULT_DEXYCB_DIR))
            c2w = _load_cam_to_world(
                dexycb_dir, src.extrinsics_id, src.camera_serial, src.camera_dir
            )

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
                # Compose cam-frame object pose with camera-to-world extrinsic.
                T_cam = np.eye(4, dtype=np.float32)
                T_cam[:3, :3] = Rcam
                T_cam[:3, 3] = tcam
                Twc = (c2w @ T_cam).astype(np.float32)
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
        """MANO kwargs paired with the pose we store.

        :func:`_expand_pca_to_aa` already bakes ``hands_mean`` into the
        45-dim axis-angle finger pose we hand off.  MANO's internal layer
        with ``flat_hand_mean=False`` would add ``hands_mean`` on top
        *again*, double-counting the mean pose and pushing fingertips up
        to ~9 cm off ground truth.  ``flat_hand_mean=True`` tells the
        layer to take our finger pose as absolute — verified joint-for-
        joint against DexYCB's ``joint_3d`` (0 mm error, vs 33 mm mean
        with ``flat_hand_mean=False``).
        """
        return {"flat_hand_mean": True, "center_idx": None}

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
