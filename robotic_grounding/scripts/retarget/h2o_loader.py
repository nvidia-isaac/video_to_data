# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load H2O dataset (Hand-Two-Object, ICCV 2021) into ManoSharpaData schema.

H2O layout (after extraction):
  h2o_dir/
    subject1/ through subject4/
      {action_category}/          e.g. h1/, k2/
        {take_id}/                e.g. 0/, 1/
          cam{0-4}/               5 cameras (cam4 is egocentric)
            hand_pose_mano/       per-frame MANO params (plain text)
              000001.txt
              000002.txt
              ...
            obj_pose_rt/          per-frame object 4x4 poses (plain text)
              000001.txt
              ...
            cam_pose/             per-frame camera poses
            rgb/ depth/           (only with --mode all; not needed for retargeting)
    object/                       object meshes

Hand pose file format (per frame, 118 floats = 2 hands * 59 floats):
  [left_hand(59), right_hand(59)]  -- ordering per official repo
  Each hand is: [flag(1), trans(3), pose(48), shape(10)]
    - flag: 1=annotated, 0=not annotated
    - trans: (tx, ty, tz) translation in camera frame
    - pose: 48 axis-angle floats = 3 global rotation + 45 finger pose
    - shape: 10 MANO betas

Object pose file format (obj_pose_rt):
  [class_id, 16 floats of 4x4 row-major transform matrix] in camera frame

Coordinate frame:
  Poses are stored in each (moving) camera's frame. H2O ships a per-frame
  ``cam_pose/{fid}.txt`` camera-to-world extrinsic; the loader composes it
  with each frame's hand and object poses so the output is in the
  gravity-aligned world frame defined by H2O's external rig calibration.

Runs stage 1 of the two-stage pipeline:
  1. python scripts/retarget/h2o_loader.py --save   -> h2o_loaded/
  2. python scripts/retarget/h2o_to_sharpa.py --save -> h2o_processed/
"""

from __future__ import annotations

import argparse
import fcntl
import tarfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mano")

from robotic_grounding.retarget import (  # noqa: E402
    ASSETS_DIR,
    HUMAN_MOTION_DATA_DIR,
    MESHES_DIR,
)
from robotic_grounding.retarget.dataset_loader_base import (  # noqa: E402
    DatasetLoaderBase,
    SequenceInfo,
    load_meshes_to_device,
    make_usd_safe,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_H2O_DIR = HUMAN_MOTION_DATA_DIR / "h2o" / "dataset"
LOADED_SAVE_DIR = HUMAN_MOTION_DATA_DIR / "h2o" / "h2o_loaded"
H2O_FPS = 30.0

# H2O object class IDs (from official dataset documentation).
# Index 0 ("background") is used when no object is active in a frame.
# TODO: verify this mapping against the actual data — in particular whether
# class IDs 1-8 are 1-indexed or 0-indexed in practice.
H2O_OBJECTS: dict[int, str] = {
    1: "book",
    2: "espresso",
    3: "lotion",
    4: "spray",
    5: "milk",
    6: "cocoa",
    7: "chips",
    8: "cappuccino",
}

# Number of floats per hand in hand_pose_mano files.
# Layout: flag(1) + trans(3) + pose(48) + shape(10) = 62 per hand, 124 total.
H2O_PER_HAND_FLOATS = 62


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class H2OSequenceSource:
    """Source info for one H2O sequence (a single camera view of a take)."""

    take_dir: Path  # points at cam{N}/ directory
    subject: str
    action_category: str
    take_id: str
    camera: str
    object_class_ids: list[int]
    object_names: list[str]


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------
def _parse_hand_pose_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Parse one hand_pose_mano text file.

    Returns:
        left_data: (59,) array for left hand
        right_data: (59,) array for right hand
        left_valid: whether left hand is annotated
        right_valid: whether right hand is annotated
    """
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.size != 2 * H2O_PER_HAND_FLOATS:
        raise ValueError(
            f"{path}: expected {2 * H2O_PER_HAND_FLOATS} floats, got {raw.size}"
        )
    left = raw[:H2O_PER_HAND_FLOATS]
    right = raw[H2O_PER_HAND_FLOATS:]
    left_valid = bool(left[0] > 0.5)
    right_valid = bool(right[0] > 0.5)
    return left, right, left_valid, right_valid


def _parse_object_pose_file(path: Path) -> tuple[int, np.ndarray]:
    """Parse one obj_pose_rt text file.

    Returns:
        class_id: integer object class ID (0 = no object).
        T: (4, 4) homogeneous transform matrix in camera frame.
    """
    raw = np.loadtxt(path, dtype=np.float32).ravel()
    if raw.size < 17:
        raise ValueError(f"{path}: expected 17 floats, got {raw.size}")
    class_id = int(raw[0])
    T = raw[1:17].reshape(4, 4)
    return class_id, T


def _parse_cam_pose_file(path: Path) -> np.ndarray:
    """Parse one cam_pose text file into a (4, 4) camera-to-world matrix."""
    raw = np.loadtxt(path, dtype=np.float32).ravel()
    if raw.size != 16:
        raise ValueError(f"{path}: expected 16 floats, got {raw.size}")
    return raw.reshape(4, 4)


def _load_cam_pose_series(cam_pose_dir: Path, frame_ids: list[str]) -> np.ndarray:
    """Return (N, 4, 4) cam-to-world series, one matrix per frame.

    H2O poses are stored in the egocentric-camera frame; the dataset's
    per-frame ``cam_pose/{fid}.txt`` is the camera→world extrinsic, and
    the world frame is gravity-aligned by construction (external rig
    calibration). If the directory or an individual file is missing, a
    warning is emitted and identity is used for those frames — keeping
    the loader runnable against H2O variants that omit cam_pose.
    """
    series = np.tile(np.eye(4, dtype=np.float32), (len(frame_ids), 1, 1))
    if not cam_pose_dir.is_dir():
        warnings.warn(
            f"cam_pose directory missing at {cam_pose_dir}; "
            "falling back to identity (output will be in camera frame).",
            stacklevel=2,
        )
        return series
    missing = 0
    for i, fid in enumerate(frame_ids):
        pose_file = cam_pose_dir / f"{fid}.txt"
        if not pose_file.exists():
            missing += 1
            continue
        series[i] = _parse_cam_pose_file(pose_file)
    if missing:
        warnings.warn(
            f"cam_pose missing for {missing}/{len(frame_ids)} frames under "
            f"{cam_pose_dir}; using identity for those frames.",
            stacklevel=2,
        )
    return series


def _collect_frame_ids(cam_dir: Path) -> list[str]:
    """Return sorted frame IDs (stems like '000001') present in the cam dir."""
    hand_dir = cam_dir / "hand_pose_mano"
    if not hand_dir.exists():
        return []
    return sorted(p.stem for p in hand_dir.glob("*.txt"))


def _extract_archives_if_needed(h2o_dir: Path) -> Path:
    """Extract H2O tarballs/zips if the dir hasn't been extracted yet.

    H2O ships as:
      subject{1,2,3,4}_pose_v1_1.tar.gz  (per-subject pose data)
      object.zip                          (object meshes)
      label_split.zip                     (annotations)

    Extraction is idempotent: if the target subject*/ directories already
    exist, we skip. Returns the directory containing the extracted data
    (same as input if already extracted, or a local cache if input is
    read-only CSS mount).
    """
    archives = sorted(h2o_dir.glob("*.tar.gz")) + sorted(h2o_dir.glob("*.zip"))
    if not archives:
        return h2o_dir  # nothing to extract; assume already-extracted data

    # Already extracted? Check if each archive has a corresponding extracted
    # directory (e.g. subject1_pose_v1_1.tar.gz -> subject1/).
    def _already_extracted(archive: Path) -> bool:
        name = archive.name
        if name.startswith("subject") and "_pose_" in name:
            return (h2o_dir / name.split("_pose_")[0]).is_dir()
        # Generic check: any dir derived from the archive name
        stem = (
            archive.stem
            if archive.suffix == ".zip"
            else archive.stem.replace(".tar", "")
        )
        return (h2o_dir / stem).is_dir()

    archives = [a for a in archives if not _already_extracted(a)]
    if not archives:
        return h2o_dir

    # Serialize extraction across shards via an advisory file lock — the
    # first shard extracts while the others block, then re-check and find
    # `_already_extracted` True.
    lock_path = Path("/tmp") / f"h2o_extract_{h2o_dir.name}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # Re-check under the lock — a sibling shard may have finished while we waited.
        archives = [a for a in archives if not _already_extracted(a)]
        if not archives:
            return h2o_dir

        # If h2o_dir is writable, extract in place. Otherwise use a cache dir.
        target = h2o_dir
        try:
            (h2o_dir / ".extract_test").touch()
            (h2o_dir / ".extract_test").unlink()
        except OSError:
            target = Path("/tmp") / f"h2o_extracted_{h2o_dir.name}"
            target.mkdir(parents=True, exist_ok=True)
            print(f"[h2o] input dir read-only, extracting to cache: {target}")

        for archive in archives:
            print(f"[h2o] extracting {archive.name} -> {target}")
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as z:
                    z.extractall(target)
            elif archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
                with tarfile.open(archive, "r:gz") as t:
                    t.extractall(target)
            else:
                print(f"[h2o] skipping unknown archive type: {archive}")

        return target


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class H2ODatasetLoader(DatasetLoaderBase):
    """Load H2O sequences into ManoSharpaData (MANO + object only)."""

    def __init__(self) -> None:
        """Initialize cached per-sequence object-id and cam-pose lookups."""
        self._args: argparse.Namespace | None = None
        # Cache object class IDs discovered per sequence so load_object_data
        # and get_object_mesh_paths can reuse the list built during
        # list_sequences / load_mano_data.
        self._object_ids_cache: dict[str, list[int]] = {}
        # Cache per-frame cam-to-world extrinsics so load_mano_data and
        # load_object_data both transform into the same world frame without
        # re-reading cam_pose/*.txt twice.
        self._cam_pose_cache: dict[str, np.ndarray] = {}

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """Discover H2O sequences (subject / action / take / camera)."""
        self._args = args
        h2o_dir = Path(getattr(args, "h2o_dir", DEFAULT_H2O_DIR))
        camera = getattr(args, "camera", "cam4")  # cam4 = egocentric

        if not h2o_dir.exists():
            raise FileNotFoundError(f"H2O dataset not found at {h2o_dir}")

        # H2O ships as tarballs; extract them the first time we see them.
        h2o_dir = _extract_archives_if_needed(h2o_dir)

        sequences: list[SequenceInfo] = []
        for subject_dir in sorted(h2o_dir.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith("subject"):
                continue
            subject = subject_dir.name
            for action_dir in sorted(subject_dir.iterdir()):
                if not action_dir.is_dir():
                    continue
                for take_dir in sorted(action_dir.iterdir()):
                    if not take_dir.is_dir():
                        continue
                    cam_dir = take_dir / camera
                    if not cam_dir.exists():
                        continue

                    sequence_id = (
                        f"h2o_{subject}_{action_dir.name}_{take_dir.name}_{camera}"
                    )
                    # Object IDs are discovered lazily in load_mano_data.
                    source = H2OSequenceSource(
                        take_dir=cam_dir,
                        subject=subject,
                        action_category=action_dir.name,
                        take_id=take_dir.name,
                        camera=camera,
                        object_class_ids=[],
                        object_names=[],
                    )
                    sequences.append(
                        SequenceInfo(
                            sequence_id=sequence_id,
                            raw_motion_file=str(cam_dir),
                            object_name="",
                            object_body_names=[],
                            source=source,
                        )
                    )

        sequences = self._apply_sequence_filters(sequences, args)
        print(f"Found {len(sequences)} H2O sequences")
        return sequences

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO hand parameters from H2O text files, one per frame."""
        src: H2OSequenceSource = sequence_info.source
        frame_ids = _collect_frame_ids(src.take_dir)
        if not frame_ids:
            raise FileNotFoundError(f"No hand pose frames in {src.take_dir}")

        right_g_list: list[np.ndarray] = []
        right_p_list: list[np.ndarray] = []
        right_t_list: list[np.ndarray] = []
        left_g_list: list[np.ndarray] = []
        left_p_list: list[np.ndarray] = []
        left_t_list: list[np.ndarray] = []
        right_betas = None
        left_betas = None

        # Track object class IDs that appear in the obj_pose_rt files so we
        # can reuse them in load_object_data and get_object_mesh_paths.
        obj_ids_seen: list[int] = []

        for fid in frame_ids:
            left_raw, right_raw, left_valid, right_valid = _parse_hand_pose_file(
                src.take_dir / "hand_pose_mano" / f"{fid}.txt"
            )

            # H2O stores a validity flag at index 0 of each per-hand vector.
            # When a hand is not annotated the remaining floats are zero/junk,
            # which fed into MANO FK + IK produces degenerate targets. Skip
            # the entire sequence rather than letting garbage frames through.
            if not (left_valid and right_valid):
                raise ValueError(
                    f"Frame {fid}: hand annotation missing "
                    f"(left_valid={left_valid}, right_valid={right_valid})"
                )

            # Split each 62-float vector: flag(1) trans(3) pose(48) shape(10)
            right_g_list.append(right_raw[4:7])
            right_p_list.append(right_raw[7:52])
            right_t_list.append(right_raw[1:4])
            left_g_list.append(left_raw[4:7])
            left_p_list.append(left_raw[7:52])
            left_t_list.append(left_raw[1:4])
            if right_betas is None:
                right_betas = right_raw[52:62].copy()
            if left_betas is None:
                left_betas = left_raw[52:62].copy()

            # Object: read the class_id from obj_pose_rt for this frame.
            obj_file = src.take_dir / "obj_pose_rt" / f"{fid}.txt"
            if obj_file.exists():
                class_id, _ = _parse_object_pose_file(obj_file)
                if class_id > 0 and class_id not in obj_ids_seen:
                    obj_ids_seen.append(class_id)

        H = len(frame_ids)
        right_global_orient = np.stack(right_g_list).astype(np.float32)
        right_finger_pose = np.stack(right_p_list).astype(np.float32)
        right_trans = np.stack(right_t_list).astype(np.float32)
        left_global_orient = np.stack(left_g_list).astype(np.float32)
        left_finger_pose = np.stack(left_p_list).astype(np.float32)
        left_trans = np.stack(left_t_list).astype(np.float32)

        # H2O stores per-frame hand/object poses in the (moving) egocentric
        # camera frame. Applying a fixed cam->world rotation leaves gravity
        # tilting with the subject's head; instead, transform each frame by
        # the per-frame cam_pose/*.txt extrinsic (camera-to-world). The
        # resulting world frame is gravity-aligned by the H2O rig calibration.
        cam_pose = _load_cam_pose_series(src.take_dir / "cam_pose", frame_ids)
        self._cam_pose_cache[sequence_info.sequence_id] = cam_pose
        for i in range(H):
            R_cw = cam_pose[i, :3, :3]
            t_cw = cam_pose[i, :3, 3]
            R_r = Rotation.from_rotvec(right_global_orient[i]).as_matrix()
            right_global_orient[i] = Rotation.from_matrix(R_cw @ R_r).as_rotvec()
            R_l = Rotation.from_rotvec(left_global_orient[i]).as_matrix()
            left_global_orient[i] = Rotation.from_matrix(R_cw @ R_l).as_rotvec()
            right_trans[i] = R_cw @ right_trans[i] + t_cw
            left_trans[i] = R_cw @ left_trans[i] + t_cw

        # Update the source with discovered objects so downstream methods can
        # fill object_body_names / mesh paths consistently.
        src.object_class_ids = obj_ids_seen
        src.object_names = [
            H2O_OBJECTS.get(cid, f"object_{cid}") for cid in obj_ids_seen
        ]
        sequence_info.object_body_names = src.object_names
        sequence_info.object_name = (
            "+".join(src.object_names) if src.object_names else "object"
        )
        self._object_ids_cache[sequence_info.sequence_id] = obj_ids_seen

        if right_betas is None or left_betas is None:
            raise ValueError(f"No MANO betas for {sequence_info.sequence_id}")

        return {
            "H": H,
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
        }

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Load per-frame object 4x4 poses from obj_pose_rt files."""
        src: H2OSequenceSource = sequence_info.source
        frame_ids = _collect_frame_ids(src.take_dir)
        obj_ids = self._object_ids_cache.get(
            sequence_info.sequence_id, src.object_class_ids
        )
        if not obj_ids:
            raise ValueError(
                f"No object class IDs for {sequence_info.sequence_id}. "
                "Did load_mano_data run first?"
            )

        # For each frame, the H2O file stores a single active object (the one
        # being manipulated).  For frames where a given object is not active,
        # we fall back to the previous known pose (and zero if never seen).
        N = len(frame_ids)

        # Use the per-frame cam-to-world extrinsic populated by load_mano_data;
        # fall back to loading directly if the cache is cold (e.g. object-only
        # invocation path).
        cam_pose = self._cam_pose_cache.get(sequence_info.sequence_id)
        if cam_pose is None:
            cam_pose = _load_cam_pose_series(src.take_dir / "cam_pose", frame_ids)

        poses = {
            cid: np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)) for cid in obj_ids
        }
        last_T: dict[int, np.ndarray] = {
            cid: np.eye(4, dtype=np.float32) for cid in obj_ids
        }
        for f_idx, fid in enumerate(frame_ids):
            obj_file = src.take_dir / "obj_pose_rt" / f"{fid}.txt"
            if not obj_file.exists():
                for cid in obj_ids:
                    poses[cid][f_idx] = last_T[cid]
                continue
            class_id, T = _parse_object_pose_file(obj_file)
            # Transform cam-frame object pose to gravity-aligned world frame
            # via this frame's cam-to-world extrinsic.
            T_world = cam_pose[f_idx] @ T
            for cid in obj_ids:
                if cid == class_id:
                    poses[cid][f_idx] = T_world
                    last_T[cid] = T_world
                else:
                    poses[cid][f_idx] = last_T[cid]

        result: dict[str, Any] = {}
        for cid in obj_ids:
            name = H2O_OBJECTS.get(cid, f"object_{cid}")
            pose_seq = poses[cid]
            root_pos = pose_seq[:, :3, 3].copy()
            root_aa = np.stack(
                [Rotation.from_matrix(T[:3, :3]).as_rotvec() for T in pose_seq]
            ).astype(np.float32)
            result[name] = (pose_seq, root_pos, root_aa, None)
        return result

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple:
        """Load OBJ meshes for each object present in this sequence.

        Searches, per object name:
        1. ``assets/meshes/h2o/{name}/*.obj`` (canonical committed location)
        2. ``{h2o_dir}/object/{name}/*.obj`` (extracted fallback from object.zip)

        Uses ``*.obj`` (glob) rather than a fixed name because H2O is
        inconsistent: e.g. the ``spray`` folder contains ``lotion_spray.obj``.

        Raises FileNotFoundError if any mesh is missing so the base class's
        sequence-level try/except can skip the sequence cleanly.
        """
        src: H2OSequenceSource = sequence_info.source
        canonical_dir = MESHES_DIR / "h2o"
        h2o_dir = Path(getattr(self._args, "h2o_dir", DEFAULT_H2O_DIR))
        fallback_dir = h2o_dir / "object"

        mesh_paths: dict[str, str] = {}
        missing: list[str] = []
        for name in src.object_names:
            candidate = None
            for base in (canonical_dir, fallback_dir):
                objs = (
                    sorted((base / name).glob("*.obj"))
                    if (base / name).is_dir()
                    else []
                )
                if objs:
                    candidate = objs[0]
                    break
            if candidate is None:
                missing.append(name)
                continue
            mesh_paths[name] = str(candidate)

        if missing:
            raise FileNotFoundError(
                f"H2O object meshes missing for {sequence_info.sequence_id}: "
                f"{missing}. Searched {canonical_dir} and {fallback_dir}."
            )
        return load_meshes_to_device(mesh_paths, device, vertex_scale=1.0)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """H2O uses standard MANO with axis-angle (no PCA, no flat hand mean)."""
        return {"flat_hand_mean": False, "center_idx": None}

    def get_fps(self) -> float:
        """H2O frame rate."""
        return H2O_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return OBJ mesh paths for all objects in the sequence.

        Mirrors the search order of :meth:`load_object_meshes` so the paths
        stored in the Parquet actually resolve at later stages
        (reconstruct / vis / video).  Searches, per object name:

        1. ``assets/meshes/h2o/{name}/*.obj`` (canonical committed copy)
        2. ``{h2o_dir}/object/{name}/*.obj`` (raw dataset fallback)

        H2O mesh filenames don't always match the folder (e.g. ``spray/``
        contains ``lotion_spray.obj``) so we glob for ``*.obj``.  If nothing
        is found, we still emit a path under the canonical dir so the
        Parquet column is well-typed — downstream consumers will then skip
        the object with a clear "no mesh loaded" warning instead of
        segfaulting.
        """
        src: H2OSequenceSource = sequence_info.source
        canonical_dir = MESHES_DIR / "h2o"
        h2o_dir = Path(getattr(self._args, "h2o_dir", DEFAULT_H2O_DIR))
        fallback_dir = h2o_dir / "object"
        paths: list[str] = []
        for name in src.object_names:
            chosen: Path | None = None
            for base in (canonical_dir, fallback_dir):
                objs = (
                    sorted((base / name).glob("*.obj"))
                    if (base / name).is_dir()
                    else []
                )
                if objs:
                    chosen = objs[0]
                    break
            paths.append(
                str(chosen) if chosen else str(canonical_dir / name / f"{name}.obj")
            )
        return paths

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return rigid URDF paths for all objects in the sequence."""
        src: H2OSequenceSource = sequence_info.source
        urdf_dir = ASSETS_DIR / "urdfs" / "h2o"
        return [
            str(urdf_dir / f"{make_usd_safe(name)}_rigid.urdf")
            for name in src.object_names
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the H2O loader script."""
    parser = argparse.ArgumentParser(
        description="Load H2O sequences into ManoSharpaData schema."
    )
    parser.add_argument(
        "--h2o_dir",
        type=Path,
        default=DEFAULT_H2O_DIR,
        help="Root directory of the H2O dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=LOADED_SAVE_DIR,
        help="Output directory for loaded Parquet files.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="cam4",
        help="Which camera view to use (default: cam4, egocentric).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument(
        "--list_sequences", action="store_true", help="List sequences and exit."
    )
    DatasetLoaderBase.add_filter_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the H2O loader."""
    loader = H2ODatasetLoader()
    if args.list_sequences:
        for s in loader.list_sequences(args):
            print(s.sequence_id)
        return
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
