# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""SOMA-X parametric body model wrapper for the SOMA-to-G1 retargeter.

Reads ``soma_params.npz`` saved by the SOMA exporter and reconstructs the
source-space joint positions, global joint orientations (wxyz), and (optional)
mesh vertices that ``WholeBodyKinematics.compute()`` consumes.

Save schema (from the SOMA exporter ``save_soma_npz``):

    poses             (T, J, 3)   per-joint local rotation vectors (rad)
    transl            (T, 3)      root translation in meters
    joint_names       (J,)        ordered SOMA rig joint names
    identity_model_type "mhr" | "soma" | ...
    identity_coeffs   (T, K_id)   identity (shape) coeffs; constant in time
    scale_params      (T, K_sc)   identity scale params; constant in time
    joint_orient      (J + 1, 3, 3) rest-pose joint orientation matrices
    unit              "meters"
    keep_root         bool        when False, ``poses[:, 0]`` is the root
    rotation_repr     "rotvec"
    absolute_pose     bool        when False, ``poses`` are local rotations

The ``SOMA`` class in this file mirrors the public surface of
the SOMA exporter so ``soma_to_g1.py`` can consume it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget import BODY_MODELS_DIR

# ---------------------------------------------------------------------------
# Lightweight container so consumers do not import torch internals.
# ---------------------------------------------------------------------------


@dataclass
class SOMAMotion:
    """Reconstructed SOMA motion in source coordinates.

    Attributes:
        joints: ``(T, J, 3)`` joint positions in meters, source frame.
        joints_wxyz: ``(T, J, 4)`` global joint rotations, wxyz quaternions.
        vertices: ``(T, V, 3)`` body mesh vertices in source frame.
        num_frames: ``T``.
        joint_names: List of SOMA rig joint names of length ``J``.
        identity_model_type: ``"mhr"``, ``"soma"`` etc.
        unit: Unit of ``transl`` from the export (always normalized to meters here).
    """

    joints: np.ndarray
    joints_wxyz: np.ndarray
    vertices: np.ndarray
    num_frames: int
    joint_names: list[str]
    identity_model_type: str
    unit: str


# ---------------------------------------------------------------------------
# SOMALayer-backed reader.
# ---------------------------------------------------------------------------


def _unit_to_meters_factor(unit: str) -> float:
    """Convert SOMA ``unit`` field to a meters scale factor."""
    u = (unit or "meters").lower()
    if u in ("m", "meter", "meters"):
        return 1.0
    if u in ("cm", "centimeter", "centimeters"):
        return 0.01
    if u in ("mm", "millimeter", "millimeters"):
        return 0.001
    raise ValueError(f"Unsupported SOMA unit {unit!r}")


_SOMA_REQUIRED_ASSETS: tuple[str, ...] = (
    "SOMA_neutral.npz",
    "correctives_model.pt",
)
_SOMA_REQUIRED_BY_IDENTITY: dict[str, tuple[str, ...]] = {
    "mhr": (
        "MHR/mhr_model_lod1.pt",
        "MHR/base_body_lod1.obj",
        "MHR/SOMA_wrap_lod1.obj",
    ),
}


def _missing_assets(root: Path, identity_model_type: str) -> list[str]:
    """Return the list of expected SOMA asset files missing under ``root``."""
    needed = list(_SOMA_REQUIRED_ASSETS) + list(
        _SOMA_REQUIRED_BY_IDENTITY.get(identity_model_type.lower(), ())
    )
    return [name for name in needed if not (root / name).is_file()]


_SETUP_HINT_PRINTED: set[tuple[str, str]] = set()


def _emit_setup_hint(
    target_root: Path,
    identity_model_type: str,
    missing: list[str],
) -> None:
    """Print a one-shot 'how to populate the SOMA cache' warning.

    Centralized so every code path that detects a missing asset (the
    explicit-data_root branch, the repo-local default branch, and any
    future ``SOMA``-style wrapper for a different robot) prints the
    *same* actionable instruction. Deduplicated per
    ``(target_root, identity_model_type)`` per process so scripts that
    construct ``SOMA`` repeatedly (e.g. tests, batch loops) do not spam
    the log. Update here only when the bootstrap contract (script name,
    default cache location) changes.
    """
    key = (str(target_root), identity_model_type)
    if key in _SETUP_HINT_PRINTED:
        return
    _SETUP_HINT_PRINTED.add(key)
    identity_flag = (
        f" --identity-model-type {identity_model_type}"
        if identity_model_type != "mhr"
        else ""
    )
    print(
        f"[read_soma] WARNING: SOMA body-model cache at {target_root} is "
        f"missing assets for identity_model_type={identity_model_type!r}: "
        f"{missing}.\n"
        f"  -> Run `python scripts/setup_soma_assets.py{identity_flag}` to "
        "download them (~822 MB from HuggingFace).\n"
        "  Falling back to SOMA-X's built-in HuggingFace cache for this "
        "run; subsequent runs will keep re-downloading until the cache "
        "above is populated."
    )


def _resolve_data_root(
    data_root: str | Path | None,
    identity_model_type: str,
) -> Path | None:
    """Resolve the SOMA assets root for ``SOMALayer``.

    Strategy (match the MANO pattern of repo-local assets, but
    never force-create an empty directory that blocks SOMA's built-in
    HuggingFace download):

    * If ``data_root`` is an explicit path and contains the full bundle
      (``SOMA_neutral.npz``, ``correctives_model.pt``, plus the
      identity-model files for ``identity_model_type``), use it.
    * If ``BODY_MODELS_DIR / "soma"`` is fully populated, use it (drop-in
      local cache alongside MANO).
    * Otherwise return ``None`` so ``SOMALayer`` auto-downloads to the
      HuggingFace cache; this keeps the first-run bootstrap working in a
      fresh Docker image without any manual asset staging. We additionally
      print a one-shot setup-script reminder so users do not silently pay
      the HuggingFace download cost on every run.

    Never create an empty ``BODY_MODELS_DIR / "soma"`` directory: SOMA-X
    treats an existing-but-incomplete directory as "assets are supposed to
    be here" and refuses to fall back to HuggingFace, which surfaces as a
    cryptic ``FileNotFoundError`` for ``SOMA_neutral.npz``.
    """
    if data_root is not None:
        path = Path(data_root).expanduser()
        missing = _missing_assets(path, identity_model_type)
        if not missing:
            return path
        _emit_setup_hint(path, identity_model_type, missing)
        return None

    repo_local = BODY_MODELS_DIR / "soma"
    if repo_local.is_dir():
        missing = _missing_assets(repo_local, identity_model_type)
        if not missing:
            return repo_local
        _emit_setup_hint(repo_local, identity_model_type, missing)
    else:
        # Fresh checkout: directory does not exist yet. Same actionable
        # guidance, treating the canonical default path as "missing
        # everything" so the user sees one consistent message.
        _emit_setup_hint(
            repo_local,
            identity_model_type,
            _missing_assets(repo_local, identity_model_type),
        )
    return None


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a stack of ``(..., 3, 3)`` matrices to wxyz quaternions."""
    flat = matrix.reshape(-1, 3, 3)
    quats_xyzw = R.from_matrix(flat).as_quat()  # scipy returns xyzw by default
    quats_wxyz = np.empty_like(quats_xyzw)
    quats_wxyz[:, 0] = quats_xyzw[:, 3]
    quats_wxyz[:, 1] = quats_xyzw[:, 0]
    quats_wxyz[:, 2] = quats_xyzw[:, 1]
    quats_wxyz[:, 3] = quats_xyzw[:, 2]
    out_shape = matrix.shape[:-2] + (4,)
    return quats_wxyz.reshape(out_shape)


class SOMA:
    """SOMA model wrapper.

    Wraps ``soma.SOMALayer`` and provides ``load_motion(...)`` returning the
    dict shape: ``joints``, ``joints_wxyz``,
    ``vertices``, ``num_frames``.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        identity_model_type: str = "mhr",
        device: torch.device | None = None,
    ) -> None:
        """Construct a ``SOMALayer`` over the local asset root.

        Args:
            data_root: Local SOMA asset directory. Defaults to
                ``assets/body_models/soma`` so SOMA-X reuses the repo asset
                layout (matching the MANO convention). The SOMA-X
                package will populate this directory on first use if empty.
            identity_model_type: SOMA identity model. ``mhr`` is the default
                used by the exporter referenced by this codebase.
            device: Torch device. Defaults to CUDA when available.
        """
        try:
            from soma import SOMALayer  # noqa: PLC0415  (lazy: heavy optional dep)
        except ImportError as exc:  # pragma: no cover - surfaced at runtime
            raise ImportError(
                "py-soma-x is required for read_soma. Install via "
                "`pip install py-soma-x` (already added to the retarget Docker image)."
            ) from exc

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.identity_model_type = identity_model_type
        self.data_root = _resolve_data_root(data_root, identity_model_type)
        self.layer = SOMALayer(
            data_root=None if self.data_root is None else str(self.data_root),
            identity_model_type=identity_model_type,
            device=str(self.device),
        )
        # ``rig_data["joint_names"]`` includes a "Root" parent entry at index
        # 0. The retargeter's joint indexing aligns with the SOMA exporter's
        # ``save_soma_npz`` convention which drops the root, so expose
        # ``rig_joint_names`` of length J without "Root".
        rig_joint_names_full = [str(n) for n in self.layer.rig_data["joint_names"]]
        self.rig_joint_names: list[str] = rig_joint_names_full[1:]
        # ``rig_data["parents"]`` does not exist; the SOMA package builds its
        # parent list from ``joint_parent_ids`` by dropping the root and
        # shifting indices, so ``self.layer.parents`` (length J) is the
        # right source. Cache as numpy for downstream FK code.
        self.parents: np.ndarray = np.asarray(self.layer.parents, dtype=np.int64)
        # Faces exposed for the SOMA visualize() method.
        # ``rig_data`` is a numpy archive (NpzFile), not a dict, so we use
        # ``.files`` to check membership rather than .get on a dict.
        rig_data = self.layer.rig_data
        rig_data_files = (
            set(rig_data.files) if hasattr(rig_data, "files") else set(rig_data)
        )
        if "triangles" in rig_data_files:
            self.faces: np.ndarray | None = np.asarray(
                rig_data["triangles"], dtype=np.int64
            )
        elif "faces" in rig_data_files:
            self.faces = np.asarray(rig_data["faces"], dtype=np.int64)
        else:
            self.faces = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_motion(
        self,
        params_path: str | Path,
        *,
        normalize: bool = False,
    ) -> dict[str, Any]:
        """Load SOMA params and return joints/orientations/vertices.

        The returned dict has the shape the retargeter
        can swap source-body wrappers without changing the per-frame loop.

        Args:
            params_path: Path to ``soma_params.npz``.
            normalize: When False (default), return the raw SOMA world-frame
                trajectory. The G1 retargeter expects raw world data because
                ``WholeBodyKinematics`` applies its own source-to-robot
                rotation and ground anchoring, mirroring the SOMA-X
                ``soma-retargeter`` reference. When True, also undo the
                frame-0 root rotation/translation so Hips sits at the
                origin; only useful for visualization / debugging.

        Returns:
            dict with keys:

            * ``joints``: ``(T, J, 3)`` joint world positions.
            * ``joints_wxyz``: ``(T, J, 4)`` global joint rotations (wxyz).
            * ``vertices``: ``(T, V, 3)`` mesh vertices.
            * ``num_frames``: ``T``.
            * ``joint_names``: SOMA rig joint name list.
            * ``identity_model_type``: ``"mhr"`` etc.
            * ``unit``: Always ``"meters"`` after normalization.
            * ``first_frame_transl``: (3,) frame-0 translation used to anchor
              the body. Always populated, even when ``normalize=False``, so
              callers can apply the same anchor to side data later.
            * ``first_frame_R_inv``: (3, 3) inverse of frame-0 root rotation.
            * ``normalized``: bool reflecting the ``normalize`` arg.
        """
        params = self._load_npz(params_path)
        meters = _unit_to_meters_factor(params["unit"])

        poses_rotvec = params["poses"].astype(np.float32)  # (T, J, 3)
        transl = params["transl"].astype(np.float32) * meters  # (T, 3)
        identity = params["identity_coeffs"][0].astype(np.float32)  # (K_id,)
        scale = params["scale_params"][0].astype(np.float32)  # (K_sc,)
        joint_orient = params["joint_orient"].astype(np.float32)  # (J + 1, 3, 3)
        keep_root = bool(np.asarray(params["keep_root"]).item())
        absolute_pose = bool(np.asarray(params.get("absolute_pose", False)).item())
        if absolute_pose:
            raise ValueError(
                "SOMA export with absolute_pose=True is not yet supported by read_soma."
            )

        T_frames = int(poses_rotvec.shape[0])

        joints, joints_wxyz, vertices = self._reconstruct(
            poses_rotvec=poses_rotvec,
            transl=transl,
            identity=identity,
            scale=scale,
            joint_orient=joint_orient,
            keep_root=keep_root,
        )

        # Compute the first-frame normalization transform so callers (e.g.
        # the object-trajectory loader in ``soma_to_g1.py``) can apply the
        # *same* transform to other world-frame data such as ``poses.npy``.
        # Use the exported root pose, not the reconstructed Hips joint frame:
        # the latter includes SOMA rig/rest-frame orientation and can tilt the
        # gravity axis, which makes rigid objects appear sideways after anchoring.
        first_frame_transform = _first_frame_transform(poses_rotvec[0, 0], transl[0])

        if normalize:
            joints, joints_wxyz, vertices = _normalize_to_first_frame(
                joints=joints,
                joints_wxyz=joints_wxyz,
                vertices=vertices,
                transform=first_frame_transform,
            )

        return {
            "joints": joints.astype(np.float64),
            "joints_wxyz": joints_wxyz.astype(np.float64),
            "vertices": vertices.astype(np.float64),
            "num_frames": T_frames,
            "joint_names": list(self.rig_joint_names),
            "identity_model_type": self.identity_model_type,
            "unit": "meters",
            # Frame-0 anchoring transform that ``normalize=True`` applied
            # (or would apply). Caller can use these to apply the same
            # anchor to side data (e.g. object trajectory):
            # ``p_anchored = (p_world - transl_first) @ R_first_inv.T``.
            "first_frame_transl": first_frame_transform["transl_first"],
            "first_frame_R_inv": first_frame_transform["R_first_inv"],
            "normalized": bool(normalize),
        }

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct(
        self,
        *,
        poses_rotvec: np.ndarray,
        transl: np.ndarray,
        identity: np.ndarray,
        scale: np.ndarray,
        joint_orient: np.ndarray,
        keep_root: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SOMALayer and return ``(joints, joints_wxyz, vertices)`` in source frame.

        Calls ``SOMALayer.forward`` to get vertices + joint world positions
        in one pass (passing ``transl`` so they land in world coords), then
        derives global joint rotations via the SOMA package's own
        ``apply_joint_orient_local`` + ``joint_local_to_world`` utilities.
        """
        del (
            joint_orient,
            keep_root,
        )  # info captured by ``self.layer.t_pose_world`` already
        T_frames = poses_rotvec.shape[0]
        J = poses_rotvec.shape[1]
        device = self.device

        with torch.no_grad():
            poses_t = torch.from_numpy(poses_rotvec).to(device)
            identity_t = (
                torch.from_numpy(identity).to(device).unsqueeze(0).expand(T_frames, -1)
            )
            scale_t = (
                torch.from_numpy(scale).to(device).unsqueeze(0).expand(T_frames, -1)
            )
            transl_t = torch.from_numpy(transl).to(device)

            output = self.layer(
                poses_t,
                identity_t,
                scale_params=scale_t,
                transl=transl_t,
            )
            verts_t = output["vertices"]
            joints_t = output["joints"]

            global_rot_t = self._compute_global_rotations(
                poses_t, T_frames=T_frames, J=J
            )

        vertices = verts_t.detach().cpu().numpy()
        joints = joints_t.detach().cpu().numpy()
        global_rot = global_rot_t.detach().cpu().numpy()

        joints_wxyz = _matrix_to_wxyz(global_rot)
        return joints, joints_wxyz, vertices

    def _compute_global_rotations(
        self,
        poses_t: torch.Tensor,
        *,
        T_frames: int,
        J: int,
    ) -> torch.Tensor:
        """Compute per-joint global rotation matrices from local rotvec poses.

        Mirrors ``SOMALayer.pose``'s rotation handling:

        * convert local rotvec poses to rotation matrices (Rodrigues),
        * pad a root identity rotation at index 0 so the array length matches
          ``rig_data["joint_names"]``,
        * apply the saved ``t_pose_world`` rest orientations via
          ``apply_joint_orient_local``,
        * propagate to world frame with ``joint_local_to_world_levelorder``.

        Returns:
            ``(T, J, 3, 3)`` global rotations in the SOMA source frame, with
            the leading root joint removed so the indexing matches the rig
            joint names exposed by ``self.rig_joint_names``.
        """
        from soma.geometry.lbs import batch_rodrigues  # noqa: PLC0415
        from soma.geometry.rig_utils import (  # noqa: PLC0415
            apply_joint_orient_local,
            compute_skeleton_levels,
            joint_local_to_world_levelorder,
            precompute_joint_orient,
        )

        device = self.device
        joint_parent_ids = self.layer.joint_parent_ids
        # ``t_pose_world`` is (J + 1, 4, 4). Use the upper 3x3 as joint_orient.
        joint_orient_world = self.layer.t_pose_world[..., :3, :3]
        orient, orient_parent_T = precompute_joint_orient(
            joint_orient_world.to(device), joint_parent_ids.to(device)
        )

        # poses_t: (T, J, 3) -> rotation matrices (T, J, 3, 3).
        local_rot = batch_rodrigues(poses_t.reshape(-1, 3)).reshape(T_frames, J, 3, 3)
        # Pad root identity to make length J + 1, matching rig_data layout.
        eye = (
            torch.eye(3, device=device, dtype=local_rot.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(T_frames, 1, 3, 3)
        )
        padded_local_rot = torch.cat([eye, local_rot], dim=1)
        oriented_local = apply_joint_orient_local(
            padded_local_rot, orient, orient_parent_T
        )
        levels = compute_skeleton_levels(joint_parent_ids, device=device)
        world_rot = joint_local_to_world_levelorder(oriented_local, levels)
        # Drop the synthetic root index 0 to match self.rig_joint_names.
        return world_rot[:, 1:, :, :]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def visualize(
        self,
        viser_server: Any,
        vertices: torch.Tensor | np.ndarray,
        root_path: str = "/soma",
        rgba: np.ndarray | None = None,
    ) -> None:
        """Visualize SOMA mesh in viser.

        Args:
            viser_server: Viser server instance.
            vertices: Mesh vertices, shape (V, 3).
            root_path: Root path in viser scene tree.
            rgba: RGBA color array, shape (4,). Defaults to skin tone.
        """
        try:
            from judo.visualizers.model import add_mesh  # noqa: PLC0415
        except ImportError:  # pragma: no cover - judo is optional in some envs
            return
        if self.faces is None:
            return

        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if rgba is None:
            rgba = np.array([255, 219, 172, 180])

        add_mesh(
            viser_server,
            f"{root_path}/mesh",
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(self.faces, dtype=np.int64),
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            rgba=rgba,
        )

    def _load_npz(self, params_path: str | Path) -> dict[str, Any]:
        """Load and lightly validate ``soma_params.npz`` keys."""
        path = Path(params_path)
        if not path.is_file():
            raise FileNotFoundError(f"SOMA params not found: {path}")
        archive = np.load(path, allow_pickle=True)
        keys = set(archive.files)
        required = {
            "poses",
            "transl",
            "joint_names",
            "identity_model_type",
            "identity_coeffs",
            "scale_params",
            "joint_orient",
            "unit",
            "keep_root",
        }
        missing = required - keys
        if missing:
            raise ValueError(
                f"SOMA params at {path} is missing fields: {sorted(missing)}; present={sorted(keys)}"
            )

        params = {k: archive[k] for k in archive.files}
        params["unit"] = (
            str(params["unit"].item())
            if params["unit"].dtype.kind == "U"
            else str(params["unit"])
        )
        identity_model_type = params["identity_model_type"]
        params["identity_model_type"] = (
            str(identity_model_type.item())
            if identity_model_type.dtype.kind == "U"
            else str(identity_model_type)
        )
        if params["identity_model_type"] != self.identity_model_type:
            raise ValueError(
                f"SOMA params identity_model_type={params['identity_model_type']!r} "
                f"does not match SOMALayer={self.identity_model_type!r}. "
                "Pass identity_model_type=<from npz> when constructing SOMA(...)."
            )
        params["joint_names"] = [str(n) for n in params["joint_names"]]
        if params["joint_names"] != self.rig_joint_names:
            raise ValueError(
                "SOMA params joint_names disagree with SOMALayer.rig_data joint_names. "
                "Re-export the sequence with the same SOMA-X version."
            )
        return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_frame_transform(
    root_rotvec_first: np.ndarray,
    transl_first: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the first-frame anchoring transform.

    The returned transform satisfies::

        p_anchored = (p_world - transl_first) @ R_first_inv.T

    for any world-frame point ``p_world``. ``root_rotvec_first`` is the
    exporter root pose at frame 0 (``poses[0, 0]`` for ``keep_root=False``).
    Do not use the reconstructed Hips joint-world orientation here: it is a
    rig joint frame, not the gravity-preserving world/root transform.
    """
    R_first = R.from_rotvec(np.asarray(root_rotvec_first, dtype=np.float64)).as_matrix()
    return {
        "transl_first": np.asarray(transl_first, dtype=np.float64),
        "R_first": R_first,
        "R_first_inv": R_first.T,
    }


def _normalize_to_first_frame(
    *,
    joints: np.ndarray,
    joints_wxyz: np.ndarray,
    vertices: np.ndarray,
    transform: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize so frame 0 is centered at the origin with canonical orientation.

    Post-processing: subtract frame-0 root
    translation and undo frame-0 root rotation across all frames so the
    downstream G1 retargeter sees a sequence anchored at the origin. This
    keeps the existing first-frame ground-anchoring logic in
    ``soma_to_g1.py``.
    """
    transl_first = transform["transl_first"]
    R_first_inv = transform["R_first_inv"]

    T_frames = joints.shape[0]
    joints_out = joints.astype(np.float64).copy()
    vertices_out = vertices.astype(np.float64).copy()
    joints_wxyz_out = joints_wxyz.astype(np.float64).copy()

    for t in range(T_frames):
        joints_out[t] = (joints_out[t] - transl_first) @ R_first_inv.T
        vertices_out[t] = (vertices_out[t] - transl_first) @ R_first_inv.T
        for j in range(joints_wxyz_out.shape[1]):
            q = joints_wxyz_out[t, j]
            mat = R.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_matrix()
            mat_corrected = R_first_inv @ mat
            xyzw = R.from_matrix(mat_corrected).as_quat()
            joints_wxyz_out[t, j] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    return joints_out, joints_wxyz_out, vertices_out


__all__ = [
    "SOMA",
    "SOMAMotion",
]
