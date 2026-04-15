# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load GRAB dataset (ECCV 2020) into ManoSharpaData schema.

Canonical GRAB layout produced by ``grab/unzip_grab.py`` from
https://github.com/otaheri/GRAB:

  grab_dir/
    grab/
      s1/ ... s10/
        {object}_{action}_{take}.npz   e.g. mug_pass_1.npz
    tools/
      object_meshes/
        contact_meshes/
          {object}.ply                 full-res GRAB object meshes (meters)
    {object}.stl                        (optional) ContactDB STLs, if the
                                        ``contactdb_scaled_stl_files_public``
                                        archive was unpacked too

Each sequence .npz contains (accessed via .item() dict):
  framerate (float, 120.0)
  n_frames (int)
  gender (str), sbj_id (str), obj_name (str)
  body.params: {transl(N,3), global_orient(N,3), body_pose(N,63),
                left_hand_pose(N,24), right_hand_pose(N,24), fullpose(N,165),
                jaw_pose(N,3), leye_pose(N,3), reye_pose(N,3), expression(N,10)}
  body.vtemp: relative path to subject rest-pose mesh
  lhand.params, rhand.params: {
      global_orient(N,3), hand_pose(N,24)  # PCA coefs,
      transl(N,3), fullpose(N,45)          # 45-DOF axis-angle finger pose
  }
  lhand.vtemp, rhand.vtemp: relative paths to per-subject hand meshes
  object.params: {global_orient(N,3), transl(N,3)}
  object.object_mesh: relative path, e.g. "tools/object_meshes/contact_meshes/airplane.ply"

GRAB does not store MANO ``betas`` or per-frame ``fitting_err`` — subject
identity is in ``vtemp`` (a rest-pose hand mesh per subject).  We use zeros
for both; the IK retarget only needs the joint positions produced by MANO
forward, and the mean-shape hand is close enough for that.

Coordinate frame:
  GRAB uses SMPL-X Y-up world.  We rotate to Z-up before storing so the
  retargeting pipeline is consistent with Arctic/OakInk2/Hot3D.

Runs stage 1 of the two-stage pipeline:
  1. python scripts/retarget/grab_loader.py --save    -> grab_loaded/
  2. python scripts/retarget/grab_to_sharpa.py --save -> grab_processed/
"""

from __future__ import annotations

import argparse
import logging
import warnings
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

logging.getLogger().setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GRAB_DIR = HUMAN_MOTION_DATA_DIR / "grab" / "dataset"
LOADED_SAVE_DIR = HUMAN_MOTION_DATA_DIR / "grab" / "grab_loaded"
GRAB_FPS = 120.0

# GRAB's SMPL-X world frame is Y-up.  Rotate to Z-up (same target as our other
# loaders) via a 90-degree rotation around X: (x, y, z) -> (x, z, -y).
# This is also the convention Arctic's loader standardises on.
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GRABSequenceSource:
    """Source info for one GRAB sequence."""

    npz_path: Path
    subject: str  # e.g. "s1"
    object_name: str  # e.g. "mug"
    action: str  # e.g. "pass"
    take: str  # e.g. "1"


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------
def _parse_sequence_name(npz_path: Path) -> tuple[str, str, str]:
    """Parse '{object}_{action}[_{take}][_Retake].npz' -> (object, action, take).

    GRAB's 51 object names are all single-word (no underscores), but take
    fields are irregular: they can be just a digit (``mug_pass_1``) or a
    digit plus a ``_Retake`` suffix (``camera_takepicture_3_Retake``), so
    splitting from the *right* misidentifies the object. Split from the
    *left* once to separate object from the rest, then again to pull off
    the action; everything remaining is the take.
    """
    stem = npz_path.stem
    head, _, rest = stem.partition("_")
    obj = head
    action, _, take = rest.partition("_")
    if not action:
        action = "unknown"
    if not take:
        take = "0"
    return obj, action, take


def _load_npz_dict(path: Path) -> dict[str, Any]:
    """Load a GRAB .npz file into a pure-Python dict.

    GRAB .npz files use numpy's object-array trick to store nested dicts.
    Each top-level key is usually a 0-d array whose .item() is a dict, so
    we resolve it here once to keep downstream code clean.
    """
    data = np.load(str(path), allow_pickle=True)
    out: dict[str, Any] = {}
    for key in data.files:
        val = data[key]
        if val.shape == () and val.dtype == object:
            out[key] = val.item()
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class GRABDatasetLoader(DatasetLoaderBase):
    """Load GRAB sequences into ManoSharpaData (MANO + object only)."""

    def __init__(self) -> None:
        """Initialize per-instance args cache populated by list_sequences."""
        self._args: argparse.Namespace | None = None

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """Discover GRAB sequences across all subjects."""
        self._args = args
        grab_dir = Path(getattr(args, "grab_dir", DEFAULT_GRAB_DIR))

        if not grab_dir.exists():
            raise FileNotFoundError(f"GRAB dataset not found at {grab_dir}")

        # The ``unzip_grab.py`` helper places per-subject data under
        # ``grab_dir/grab/sN/``.  Support both that canonical layout and the
        # shorter ``grab_dir/sN/`` in case someone flattens it by hand.
        subjects_root = grab_dir / "grab" if (grab_dir / "grab").is_dir() else grab_dir

        sequences: list[SequenceInfo] = []
        for subject_dir in sorted(subjects_root.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith("s"):
                continue
            if not subject_dir.name[1:].isdigit():
                continue
            subject = subject_dir.name
            for npz_path in sorted(subject_dir.glob("*.npz")):
                obj, action, take = _parse_sequence_name(npz_path)
                sequence_id = f"grab_{subject}_{obj}_{action}_{take}"
                source = GRABSequenceSource(
                    npz_path=npz_path,
                    subject=subject,
                    object_name=obj,
                    action=action,
                    take=take,
                )
                sequences.append(
                    SequenceInfo(
                        sequence_id=sequence_id,
                        raw_motion_file=str(npz_path),
                        object_name=obj,
                        object_body_names=[obj],
                        source=source,
                    )
                )

        sequences = self._apply_sequence_filters(sequences, args)
        print(f"Found {len(sequences)} GRAB sequences")
        return sequences

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO hand parameters from a GRAB .npz file.

        Applies Y-up -> Z-up rotation so downstream code sees a consistent
        world frame.  With ``center_idx=None``, the MANO wrist world position
        is ``R_global @ J_shaped[0](betas) + transl``.  Rotating the world
        requires the betas-aware ``new_transl`` transform (same pattern as
        hot3d_loader.py).
        """
        src: GRABSequenceSource = sequence_info.source
        data = _load_npz_dict(src.npz_path)

        n_frames = int(data.get("n_frames", 0))
        fps = int(data.get("framerate", GRAB_FPS))
        if fps != GRAB_FPS:
            print(
                f"Warning: {src.npz_path.name} framerate={fps}, expected {int(GRAB_FPS)}"
            )

        rhand = data["rhand"]["params"]
        lhand = data["lhand"]["params"]

        # GRAB stores finger pose two ways: ``hand_pose`` as 24-dim PCA coefs
        # and ``fullpose`` as the already-expanded 45-dim axis-angle.  We use
        # ``fullpose`` to avoid an extra PCA->axis-angle expansion step.
        right_global = np.asarray(rhand["global_orient"], dtype=np.float32)
        right_pose = np.asarray(rhand["fullpose"], dtype=np.float32)
        right_trans = np.asarray(rhand["transl"], dtype=np.float32)

        left_global = np.asarray(lhand["global_orient"], dtype=np.float32)
        left_pose = np.asarray(lhand["fullpose"], dtype=np.float32)
        left_trans = np.asarray(lhand["transl"], dtype=np.float32)

        # GRAB has no per-hand betas (subject identity is in vtemp).  Use the
        # mean-shape MANO (zeros) — the IK retarget tracks joint positions, so
        # small shape mismatch is acceptable.
        right_betas = np.zeros(10, dtype=np.float32)
        left_betas = np.zeros(10, dtype=np.float32)

        # No fitting error is stored.
        right_fit = np.zeros(n_frames, dtype=np.float32)
        left_fit = np.zeros(n_frames, dtype=np.float32)

        # Apply Y-up -> Z-up rotation to global orient + translation.
        # manotorch (center_idx=None) computes wrist_world = R_g @ J0 + trans,
        # so rotating the world requires: new_trans = R @ (J0 + trans) - J0.
        # We don't have J0 here without running MANO FK; use the zero-pose
        # wrist offset computed from betas (same as hot3d_loader.py).
        from manotorch.manolayer import ManoLayer  # noqa: E402, PLC0415

        R_world = Y_UP_TO_Z_UP

        def _zero_pose_joint0(side: str, betas: np.ndarray) -> np.ndarray:
            layer = ManoLayer(
                mano_assets_root=str(ASSETS_DIR / "body_models" / "mano"),
                side=side,
                flat_hand_mean=False,
                use_pca=True,
                ncomps=15,
                center_idx=None,
            )
            with torch.no_grad():
                out = layer(
                    pose_coeffs=torch.zeros(1, 3 + 15),
                    betas=torch.from_numpy(betas).float().unsqueeze(0),
                )
            return out.joints[0, 0].cpu().numpy()

        try:
            J_right = _zero_pose_joint0("right", right_betas)
            J_left = _zero_pose_joint0("left", left_betas)
        except Exception:
            # If MANO assets aren't available (local dev), fall back to no-offset.
            # This slightly mis-aligns the wrist after rotation but keeps iteration going.
            J_right = np.zeros(3, dtype=np.float32)
            J_left = np.zeros(3, dtype=np.float32)

        for i in range(n_frames):
            M = Rotation.from_rotvec(right_global[i]).as_matrix()
            right_global[i] = Rotation.from_matrix(R_world @ M).as_rotvec()
            M = Rotation.from_rotvec(left_global[i]).as_matrix()
            left_global[i] = Rotation.from_matrix(R_world @ M).as_rotvec()

        right_trans = (R_world @ (right_trans + J_right).T).T - J_right
        left_trans = (R_world @ (left_trans + J_left).T).T - J_left

        return {
            "H": n_frames,
            "right_global_orient": torch.from_numpy(right_global).to(device),
            "right_finger_pose": torch.from_numpy(right_pose).to(device),
            "right_trans": torch.from_numpy(right_trans).to(device),
            "right_betas": torch.from_numpy(right_betas).to(device),
            "right_fitting_err": torch.from_numpy(right_fit).to(device),
            "left_global_orient": torch.from_numpy(left_global).to(device),
            "left_finger_pose": torch.from_numpy(left_pose).to(device),
            "left_trans": torch.from_numpy(left_trans).to(device),
            "left_betas": torch.from_numpy(left_betas).to(device),
            "left_fitting_err": torch.from_numpy(left_fit).to(device),
        }

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Load object 4x4 poses from a GRAB .npz file (single rigid object)."""
        src: GRABSequenceSource = sequence_info.source
        data = _load_npz_dict(src.npz_path)

        obj_params = data["object"]["params"]
        obj_rotvec = np.asarray(obj_params["global_orient"], dtype=np.float32)
        obj_trans = np.asarray(obj_params["transl"], dtype=np.float32)
        n_frames = obj_rotvec.shape[0]

        poses = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
        for i in range(n_frames):
            R_local = Rotation.from_rotvec(obj_rotvec[i]).as_matrix()
            # Rotate to Z-up: new_R = R_world @ R_local, new_t = R_world @ t
            poses[i, :3, :3] = Y_UP_TO_Z_UP @ R_local
            poses[i, :3, 3] = Y_UP_TO_Z_UP @ obj_trans[i]

        root_position = poses[:, :3, 3].copy()
        root_axis_angle = np.stack(
            [Rotation.from_matrix(T[:3, :3]).as_rotvec() for T in poses]
        ).astype(np.float32)

        return {src.object_name: (poses, root_position, root_axis_angle, None)}

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple:
        """Resolve the GRAB object mesh for this sequence.

        GRAB's canonical object meshes live at
        ``{grab_dir}/tools/object_meshes/contact_meshes/{name}.ply``.  We also
        look at a committed copy under ``assets/meshes/grab/`` and (as a last
        resort) at the flat ContactDB STL files at ``{grab_dir}/{name}.stl``
        — note the ContactDB names use underscores (``alarm_clock.stl``) while
        GRAB npz/ply names don't (``alarmclock.ply``), so this fallback only
        matches for the no-underscore objects.
        """
        src: GRABSequenceSource = sequence_info.source
        grab_dir = Path(getattr(self._args, "grab_dir", DEFAULT_GRAB_DIR))
        canonical_dir = MESHES_DIR / "grab"
        contact_meshes_dir = grab_dir / "tools" / "object_meshes" / "contact_meshes"

        mesh_paths: dict[str, str] = {}
        missing: list[str] = []
        for name in [src.object_name]:
            chosen: Path | None = None
            candidates = [
                canonical_dir / name / "mesh_tex.obj",
                canonical_dir / f"{name}.ply",
                canonical_dir / f"{name}.stl",
                contact_meshes_dir / f"{name}.ply",
                grab_dir / f"{name}.stl",
            ]
            for c in candidates:
                if c.exists():
                    chosen = c
                    break
            if chosen is None:
                missing.append(name)
                continue
            mesh_paths[name] = str(chosen)

        if missing:
            raise FileNotFoundError(
                f"GRAB object meshes missing for {sequence_info.sequence_id}: "
                f"{missing}. Searched {canonical_dir} and {contact_meshes_dir}."
            )
        return load_meshes_to_device(mesh_paths, device, vertex_scale=1.0)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """GRAB uses standard MANO axis-angle (no PCA) without joint centering."""
        return {"flat_hand_mean": False, "center_idx": None}

    def get_fps(self) -> float:
        """GRAB native frame rate."""
        return GRAB_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return the mesh path that load_object_meshes would pick.

        Kept in sync with the candidate list in ``load_object_meshes``.
        """
        src: GRABSequenceSource = sequence_info.source
        grab_dir = Path(getattr(self._args, "grab_dir", DEFAULT_GRAB_DIR))
        canonical_dir = MESHES_DIR / "grab"
        contact_meshes_dir = grab_dir / "tools" / "object_meshes" / "contact_meshes"
        name = src.object_name
        candidates = [
            canonical_dir / name / "mesh_tex.obj",
            canonical_dir / f"{name}.ply",
            canonical_dir / f"{name}.stl",
            contact_meshes_dir / f"{name}.ply",
            grab_dir / f"{name}.stl",
        ]
        for c in candidates:
            if c.exists():
                return [str(c)]
        return [str(candidates[0])]  # path stub for downstream error reporting

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return rigid URDF path for the sequence's single object."""
        src: GRABSequenceSource = sequence_info.source
        urdf_dir = ASSETS_DIR / "urdfs" / "grab"
        return [str(urdf_dir / f"{make_usd_safe(src.object_name)}_rigid.urdf")]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GRAB loader script."""
    parser = argparse.ArgumentParser(
        description="Load GRAB sequences into ManoSharpaData schema."
    )
    parser.add_argument(
        "--grab_dir",
        type=Path,
        default=DEFAULT_GRAB_DIR,
        help="Root directory of the extracted GRAB dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=LOADED_SAVE_DIR,
        help="Output directory for loaded Parquet files.",
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
    """Run the GRAB loader."""
    loader = GRABDatasetLoader()
    if args.list_sequences:
        for s in loader.list_sequences(args):
            print(s.sequence_id)
        return
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
