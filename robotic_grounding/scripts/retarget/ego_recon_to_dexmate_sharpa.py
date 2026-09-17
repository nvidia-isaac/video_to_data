# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retarget an ego_recon reconstruction bundle to the fixed-base Dexmate/Vega Sharpa robot.

Reads ``result.npz`` plus the loader's world-space MANO Parquet, places the scene into
the robot workspace, solves per-frame IK via
``DexmateSharpaIK`` (``mano_to_dexmate_sharpa.py``), and writes:

- a ``motion_v1`` ``single_robot`` parquet at ``--target_fps``, and
- a native-fps diagnostics ``.npz`` sidecar with the raw IK solution and targets.

World-frame contract: all parquet world quantities (object poses, ``ee_pose_w``) are in
the REDUCED-model world (``vega_sharpa_reduced.urdf`` with root at the origin) — the
gravity-aligned placement world, and the model ``ROBOT_REGISTRY['vega_sharpa']``
spawns for replay. ``T_reduced_to_fixed`` is NOT applied to world poses: it is not a pure
yaw (the fixed URDF zeroes the reduced URDF's frozen ~20 deg torso pitch), so pushing it
into world poses tilts gravity and offsets the scene from the replayed robot. Joint names
are shared between the two URDFs; the fixed-order remap ships in the sidecar only.

Run inside the retarget container (host python lacks pink/pinocchio):
    python scripts/retarget/ego_recon_to_dexmate_sharpa.py \
        --loaded_dir /workspace/e2e/intermediate/ego_recon/loaded --save
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_SCRIPT_DIR = Path(__file__).resolve().parent
RG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
RG_SOURCE = RG_ROOT / "source" / "robotic_grounding"
MANO_LIB_DIR = (
    REPO_ROOT / "reconstruction" / "modules" / "v2d_task_library_loader" / "lib"
)
IK_SCRIPT = _SCRIPT_DIR / "mano_to_dexmate_sharpa.py"

sys.path.insert(0, str(RG_SOURCE))
sys.path.insert(0, str(MANO_LIB_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

import contact_labels  # noqa: E402  (sibling module in scripts/retarget)
from robotic_grounding.motion_schema import (  # noqa: E402
    MotionData,
    save_motion_parquet,
)
from robotic_grounding.retarget import ASSETS_DIR  # noqa: E402
from robotic_grounding.retarget.joint_limits import (  # noqa: E402
    clamp_position_limits,
)

DEFAULT_BUNDLE_DIR = (
    REPO_ROOT / "reconstruction" / "data" / "tissue_box" / "result_leveled"
)
DEFAULT_OUTPUT_ROOT = ASSETS_DIR / "human_motion_data" / "ego_recon" / "processed"
# QA artifacts (verification video, IK sidecar, report) are regenerable and must not
# land in the committed asset tree alongside the parquet and object meshes.
DEFAULT_ARTIFACT_DIR = RG_ROOT / "out"
FIXED_URDF = ASSETS_DIR / "urdfs" / "vega_sharpa" / "vega_sharpa_reduced_fixed.urdf"
CONTAINER_REPO_ROOT = Path("/workspace/video_to_data")

# Dexmate/Vega whole-body embodiment. Distinct from the floating-hand `sharpa_wave`
# retarget (ego_recon_to_sharpa.py), which writes its own parquet for the same clip.
ROBOT_NAME = "vega_sharpa"
MOTION_KIND = "single_robot"
COORD_FRAME = "robot_base_z_up"
SOURCE_DATASET = "ego_recon"
EE_LINK_NAMES = ["L_arm_l7", "R_arm_l7"]
PARQUET_FILE_NAME = "data.parquet"

# Wrist frame offset (Sharpa hand_C_MC vs MANO wrist). Post-multiplied INVERTED, matching
# hand_kinematics.SharpaHandKinematics (target_rot @ R(offset).inv()) and
# validate_mano_to_dexmate_sharpa's wrist convention.
WRIST_OFFSET_WXYZ = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float64)
# Fingertip site->URDF correction. Post-multiplied AS-IS, matching
# validate_mano_to_dexmate_sharpa.TIP_CORRECTION_WXYZ (transform @ correction).
TIP_CORRECTION_WXYZ = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float64)

# Interior margin (rad) applied when clamping output joints to the URDF limits. The IK
# clamps exactly onto limits (margin ~1e-16), and the float32 cast in the parquet can land
# an epsilon OUTSIDE them, which Isaac Lab's articulation _validate_cfg rejects strictly.
JOINT_LIMIT_MARGIN_RAD = 1e-4
LOADED_WORLD_ALIGNMENT_TOL = 1e-4

# MANO_JOINTS_ORDER index -> IK frame-name suffix. Tips at 4/8/12/16/20.
MANO_INDEX_BY_TARGET_SUFFIX: dict[str, int] = {
    "hand_C_MC": 0,
    "thumb_MCP_VL": 2,
    "thumb_fingertip": 4,
    "index_MP": 5,
    "index_fingertip": 8,
    "middle_MP": 9,
    "middle_fingertip": 12,
    "ring_MP": 13,
    "ring_fingertip": 16,
    "pinky_MP": 17,
    "pinky_fingertip": 20,
}

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Robot fingertip link frames used as hand keypoints for the tracking reward. These are
# the distal-phalanx (``*_DP``) bodies that ``TrackingCommand`` matches on
# (``fingertip_body_name=".*_DP"``); the retargeted keypoint MUST be the DP frame so it
# aligns with the achieved live-sim fingertip body pose the reward compares against.
HAND_KEYPOINT_FRAMES: dict[str, list[str]] = {
    side: [f"{side}_{finger}_DP" for finger in FINGERS] for side in ("left", "right")
}

# IK task-frame origins used as contact probes (11 per side, task order).
_CONTACT_PROBE_SUFFIXES = (
    "hand_C_MC",
    "thumb_MCP_VL",
    "thumb_fingertip",
    "index_MP",
    "index_fingertip",
    "middle_MP",
    "middle_fingertip",
    "ring_MP",
    "ring_fingertip",
    "pinky_MP",
    "pinky_fingertip",
)
CONTACT_PROBE_FRAMES: dict[str, list[str]] = {
    side: [f"{side}_{suffix}" for suffix in _CONTACT_PROBE_SUFFIXES]
    for side in ("left", "right")
}

# Contact-generation defaults.
DEFAULT_CONTACT_THRESHOLD_M = 0.018
DEFAULT_MIN_CONSECUTIVE = 2

# Frames whose difference gives the reduced-world -> fixed-world rigid transform.
T_R2F_CANDIDATE_FRAMES = ("torso_l3", "base")

# Suffix of the rigid URDF that `generate_rigid_urdfs.py` writes beside the object mesh;
# `--reuse_object_assets` resolves that pair instead of writing its own scaled copy.
REUSED_URDF_SUFFIX = "_rigid.urdf"

OBJECT_MASS_KG = 0.3
NUM_MANO_JOINTS = 21
RESAMPLE_FPS_EPSILON = 1e-6
PERCENTILE_95 = 95.0
# Bundle object_to_world 3x3 blocks may embed a uniform scale (e.g. 0.9433 in the
# default bundle). It must be near-constant across frames; warn when it is far from 1.
EMBEDDED_SCALE_SPREAD_TOL = 1e-3
EMBEDDED_SCALE_WARN_TOL = 0.01


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def to_container_path(path: Path) -> str:
    """Map a repo path to its in-container absolute path (bind mount at /workspace)."""
    resolved = path.resolve()
    try:
        return str(CONTAINER_REPO_ROOT / resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def orthonormalize_rotations(rotations: np.ndarray) -> np.ndarray:
    """Project a batch of (possibly scaled) 3x3 blocks onto SO(3) via SVD.

    Args:
        rotations: Array of shape (T, 3, 3).

    Returns:
        Array of shape (T, 3, 3) of proper rotation matrices.
    """
    u, _s, vt = np.linalg.svd(rotations.astype(np.float64))
    det = np.linalg.det(np.einsum("tij,tjk->tik", u, vt))
    u = u.copy()
    u[det < 0.0, :, 2] *= -1.0
    return np.einsum("tij,tjk->tik", u, vt)


def matrices_to_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert a batch of rotation matrices (..., 3, 3) to wxyz quaternions (..., 4)."""
    shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3)
    quats = Rotation.from_matrix(flat).as_quat(scalar_first=True)
    return quats.reshape(*shape, 4)


def wxyz_to_matrices(wxyz: np.ndarray) -> np.ndarray:
    """Convert a batch of wxyz quaternions (..., 4) to rotation matrices (..., 3, 3)."""
    shape = wxyz.shape[:-1]
    flat = np.asarray(wxyz, dtype=np.float64).reshape(-1, 4)
    matrices = Rotation.from_quat(flat, scalar_first=True).as_matrix()
    return matrices.reshape(*shape, 3, 3)


def pose_to_matrix(position: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    """Build one homogeneous transform from a position and a wxyz quaternion."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def error_stats(
    values: np.ndarray, frame_indices: np.ndarray
) -> dict[str, float | int]:
    """Summarize an error array as mean/p95/max plus the worst source frame index."""
    per_frame_max = (
        values.max(axis=tuple(range(1, values.ndim))) if values.ndim > 1 else values
    )
    worst = int(frame_indices[int(np.argmax(per_frame_max))])
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, PERCENTILE_95)),
        "max": float(values.max()),
        "worst_frame": worst,
    }


def load_ik_module(path: Path = IK_SCRIPT) -> ModuleType:
    """Load the DexmateSharpaIK implementation from its file path."""
    spec = importlib.util.spec_from_file_location("mano_to_dexmate_sharpa", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the IK module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Bundle loading and MANO world tracks
# ---------------------------------------------------------------------------


def load_bundle(
    bundle_dir: Path, frame_start: int, max_frames: int | None, warnings: list[str]
) -> dict[str, np.ndarray]:
    """Load ``result.npz``, assert gravity alignment, and slice the frame range.

    Frames flagged invalid by the per-track ``*_is_valid`` masks are appended to
    ``warnings`` (invalid frames typically hold finite stale/identity poses that
    would otherwise silently feed garbage IK targets).
    """
    npz_path = bundle_dir / "result.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"result.npz not found: {npz_path}")
    npz = np.load(npz_path)

    if not bool(npz["gravity_alignment_applied"]):
        raise ValueError(
            f"{npz_path} has gravity_alignment_applied=False; this driver requires "
            "gravity-aligned world tracks."
        )
    print("gravity_alignment_applied=True (world tracks are already gravity aligned)")

    total = int(npz["camera_to_world_transform"].shape[0])
    end = total if max_frames is None else min(total, frame_start + max_frames)
    if not 0 <= frame_start < end:
        raise ValueError(f"Empty frame range [{frame_start}, {end}) for {total} frames")
    sl = slice(frame_start, end)

    data: dict[str, np.ndarray] = {
        "camera_to_world": npz["camera_to_world_transform"][sl].astype(np.float64),
        "object_to_world": npz["object_to_world_transform"][sl].astype(np.float64),
    }
    for side in ("left", "right"):
        data[f"{side}_betas"] = npz[f"hand_{side}_betas"].astype(np.float64)
        data[f"{side}_orient_cam"] = npz[f"hand_{side}_wrist_orient_in_camera"][
            sl
        ].astype(np.float64)
        data[f"{side}_finger_pose"] = npz[f"hand_{side}_finger_pose"][sl].astype(
            np.float64
        )
        data[f"{side}_wrist_to_world"] = npz[f"hand_{side}_wrist_to_world_transform"][
            sl
        ].astype(np.float64)
        data[f"{side}_scale"] = npz[f"hand_{side}_scale"][sl].astype(np.float64)
    data["object_scale"] = np.asarray(
        npz["object_scale"] if "object_scale" in npz.files else 1.0, dtype=np.float64
    )

    for key in ("camera", "hand_left", "hand_right", "object"):
        mask_key = f"{key}_is_valid"
        if mask_key not in npz.files:
            continue
        mask = npz[mask_key][sl].astype(bool)
        if not mask.all():
            invalid = (np.nonzero(~mask)[0] + frame_start).tolist()
            warnings.append(
                f"{mask_key}=False on {len(invalid)} sliced frames (source indices "
                f"{invalid}); their stored poses are stale/untrusted."
            )
            print(f"WARNING: {warnings[-1]}")
    npz.close()

    for key, value in data.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite values in bundle array {key!r}")
    return data


def correct_left_hand_params(
    orient_cam: np.ndarray, finger_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror-correct WiLoR left-hand params stored in right-hand parameter space.

    WiLoR/HaMeR store LEFT-hand MANO params in RIGHT-hand space; manotorch's native
    left layer expects native-left params, so negate the y,z axis-angle components of
    the global orient and every one of the 15 finger joints (ego_recon_loader pattern).
    """
    orient = orient_cam.copy()
    orient[:, 1:] *= -1.0
    fingers = finger_pose.copy()
    fingers[:, :, 1:] *= -1.0
    return orient, fingers


def mano_camera_fk(
    mano: Any,
    side: str,
    betas: np.ndarray,
    orient_cam: np.ndarray,
    finger_pose: np.ndarray,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MANO FK in the camera frame with zero translation.

    Returns:
        Tuple of joints (T, 21, 3) and joints_wxyz (T, 21, 4) in MANO_JOINTS_ORDER.
    """
    import torch  # noqa: PLC0415

    num_frames = orient_cam.shape[0]
    result = mano.forward(
        side=side,
        betas=torch.from_numpy(betas).float().to(device),
        global_orient=torch.from_numpy(orient_cam).float().to(device),
        transl=torch.zeros((num_frames, 3), dtype=torch.float32, device=device),
        finger_pose=torch.from_numpy(finger_pose.reshape(num_frames, 45))
        .float()
        .to(device),
    )
    joints = result["joints"].detach().cpu().numpy().astype(np.float64)
    joints_wxyz = result["joints_wxyz"].detach().cpu().numpy().astype(np.float64)
    if joints.shape != (num_frames, NUM_MANO_JOINTS, 3):
        raise ValueError(f"Unexpected MANO joints shape {joints.shape}")
    return joints, joints_wxyz


def hand_world_tracks(
    joints_cam: np.ndarray,
    joints_wxyz_cam: np.ndarray,
    camera_to_world: np.ndarray,
    wrist_to_world: np.ndarray,
    scene_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose scaled world hand tracks from camera-frame MANO FK.

    The stored wrist transform translation is the MANO ``transl`` anchor (WiLoR
    ``cam_t``), NOT the wrist joint position: with ``center_idx=None`` the camera-frame
    FK at ``transl=0`` keeps the rest-root offset J0 in ``joints_cam``, and the
    authoritative ``ego_recon_loader`` FK gate enforces
    ``joints_world = R_c2w @ joints_cam + t_stored``. That composition is used here
    with only the stored world translation multiplied by ``scene_scale``; the hand
    geometry (finger articulation AND the metric J0 offset carried by ``joints_cam``)
    deliberately stays metric. Rotations are recomposed from the (mirror-corrected)
    camera-frame orientations: R_w = R_c2w @ R_cam.

    Returns:
        Tuple of joint positions (T, 21, 3) and rotation matrices (T, 21, 3, 3).
    """
    rotation_c2w = camera_to_world[:, :3, :3]
    anchor_w = scene_scale * wrist_to_world[:, :3, 3]
    joints_w = (
        np.einsum("tij,tkj->tki", rotation_c2w, joints_cam) + anchor_w[:, None, :]
    )
    rotations_cam = wxyz_to_matrices(joints_wxyz_cam)
    rotations_w = np.einsum("tij,tkjl->tkil", rotation_c2w, rotations_cam)
    return joints_w, rotations_w


def reconcile_loaded_world_tracks(
    joints_by_side: dict[str, np.ndarray],
    object_position_loaded: np.ndarray,
    object_rotation_loaded: np.ndarray,
    data: dict[str, np.ndarray],
    scene_scale: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Restore the direct-route scale contract after loader world re-posing.

    The loader is invoked with ``--no_ground_align`` and therefore may translate the
    whole scene to center the object, but it must not rotate it. Remove that common
    translation, preserve metric MANO geometry, and scale only the reconstructed world
    anchors exactly as :func:`hand_world_tracks` does in the legacy direct route.
    """
    raw_object_position = data["object_to_world"][:, :3, 3]
    raw_object_rotation = orthonormalize_rotations(data["object_to_world"][:, :3, :3])
    if object_position_loaded.shape != raw_object_position.shape:
        raise ValueError(
            "Loaded object positions do not match the reconstruction frame count: "
            f"{object_position_loaded.shape} != {raw_object_position.shape}"
        )
    if object_rotation_loaded.shape != raw_object_rotation.shape:
        raise ValueError(
            "Loaded object rotations do not match the reconstruction frame count: "
            f"{object_rotation_loaded.shape} != {raw_object_rotation.shape}"
        )

    translation_per_frame = object_position_loaded - raw_object_position
    loader_translation = translation_per_frame.mean(axis=0)
    translation_error = float(
        np.max(np.abs(translation_per_frame - loader_translation))
    )
    rotation_error = float(np.max(np.abs(object_rotation_loaded - raw_object_rotation)))
    if (
        translation_error > LOADED_WORLD_ALIGNMENT_TOL
        or rotation_error > LOADED_WORLD_ALIGNMENT_TOL
    ):
        raise ValueError(
            "Loader-produced tracks changed the gravity-aligned world rotation or "
            "applied a non-rigid translation. Re-run the loader with "
            f"--no_ground_align (translation_error={translation_error:.3e}, "
            f"rotation_error={rotation_error:.3e})."
        )

    adjusted_joints: dict[str, np.ndarray] = {}
    for side, joints in joints_by_side.items():
        anchor = data[f"{side}_wrist_to_world"][:, :3, 3]
        if joints.shape[0] != anchor.shape[0]:
            raise ValueError(
                f"Loaded {side} MANO frames do not match wrist anchors: "
                f"{joints.shape[0]} != {anchor.shape[0]}"
            )
        adjusted_joints[side] = (
            joints
            - loader_translation[None, None, :]
            + (scene_scale - 1.0) * anchor[:, None, :]
        )

    scaled_object_position = scene_scale * (
        object_position_loaded - loader_translation[None, :]
    )
    return adjusted_joints, scaled_object_position, object_rotation_loaded


# ---------------------------------------------------------------------------
# Placement and IK targets
# ---------------------------------------------------------------------------


def compute_placement(
    ik: Any,
    left_wrist_w: np.ndarray,
    right_wrist_w: np.ndarray,
    yaw_deg: float,
    extra_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the scene->robot placement [R_place | t_place].

    ``t_place`` maps the mean scene wrist midpoint onto the midpoint of the IK
    reference-posture wrists, plus an operator-supplied extra offset.
    """
    reference = ik.fk(ik.q_reference, ("left_hand_C_MC", "right_hand_C_MC"))
    wrist_mid_ref = 0.5 * (
        np.asarray(reference["left_hand_C_MC"].position, dtype=np.float64)
        + np.asarray(reference["right_hand_C_MC"].position, dtype=np.float64)
    )
    wrist_mid_scene = 0.5 * (left_wrist_w.mean(axis=0) + right_wrist_w.mean(axis=0))
    r_place = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    t_place = wrist_mid_ref - r_place @ wrist_mid_scene + extra_offset
    return r_place, t_place


def build_frame_targets(
    ik_module: ModuleType,
    task_frame_names: tuple[str, ...],
    placed_joints: dict[str, np.ndarray],
    placed_rotations: dict[str, np.ndarray],
    frame_idx: int,
    mano_to_robot_scale: float,
) -> dict[str, Any]:
    """Build the full 22-frame target dict for one solve.

    Positions: wrist as-is; all other targets scaled about the wrist by
    ``mano_to_robot_scale``. Orientations: wrist post-multiplies the inverted
    Sharpa wrist offset; fingertips post-multiply the tip correction; MP frames
    (orientation cost 0) use the raw placed MANO rotation.
    """
    wrist_offset_inv = (
        Rotation.from_quat(WRIST_OFFSET_WXYZ, scalar_first=True).inv().as_matrix()
    )
    tip_correction = Rotation.from_quat(
        TIP_CORRECTION_WXYZ, scalar_first=True
    ).as_matrix()

    targets: dict[str, Any] = {}
    for name in task_frame_names:
        side, suffix = name.split("_", 1)
        mano_idx = MANO_INDEX_BY_TARGET_SUFFIX[suffix]
        joints = placed_joints[side][frame_idx]
        rotations = placed_rotations[side][frame_idx]

        wrist_position = joints[MANO_INDEX_BY_TARGET_SUFFIX["hand_C_MC"]]
        if suffix == "hand_C_MC":
            position = wrist_position
            rotation = rotations[mano_idx] @ wrist_offset_inv
        else:
            position = (
                wrist_position
                + (joints[mano_idx] - wrist_position) * mano_to_robot_scale
            )
            if suffix.endswith("fingertip"):
                rotation = rotations[mano_idx] @ tip_correction
            else:
                rotation = rotations[mano_idx]
        targets[name] = ik_module.FrameTarget(
            position=position,
            wxyz=Rotation.from_matrix(rotation).as_quat(scalar_first=True),
        )
    return targets


# ---------------------------------------------------------------------------
# Fixed-model remap, resampling, and object assets
# ---------------------------------------------------------------------------


def build_fixed_model() -> tuple[Any, Any]:
    """Build the pinocchio model + data for the fixed-orientation URDF."""
    import pinocchio as pin  # noqa: PLC0415

    model = pin.buildModelFromUrdf(str(FIXED_URDF))
    data = model.createData()
    return model, data


def joint_q_indices(model: Any) -> dict[str, int]:
    """Map each movable joint name to its q index (requires nq == nv, 1-DoF joints)."""
    if model.nq != model.nv:
        raise ValueError(f"Expected nq == nv, got nq={model.nq}, nv={model.nv}")
    return {
        str(name): int(model.idx_qs[model.getJointId(str(name))])
        for name in model.names[1:]
    }


def clamp_q_to_limits(q: np.ndarray, model: Any, margin: float) -> np.ndarray:
    """Clamp a (T, nq) trajectory strictly inside the model's position limits.

    Args:
        q: Joint trajectory in the model's q order.
        model: Pinocchio model providing lower/upper position limits.
        margin: Interior margin in radians; collapsed to the interval midpoint when a
            joint's range is narrower than twice the margin.

    Returns:
        The clamped copy of ``q``.
    """
    return clamp_position_limits(
        q,
        model.lowerPositionLimit,
        model.upperPositionLimit,
        margin,
    )


def remap_q_by_name(
    q_source: np.ndarray, source_indices: dict[str, int], target_indices: dict[str, int]
) -> np.ndarray:
    """Remap a (T, nq) trajectory between two same-joint-set pinocchio q orders."""
    if set(source_indices) != set(target_indices):
        raise ValueError(
            "Joint name mismatch between models: "
            f"only_source={sorted(set(source_indices) - set(target_indices))}, "
            f"only_target={sorted(set(target_indices) - set(source_indices))}"
        )
    q_target = np.zeros_like(q_source)
    for name, src_idx in source_indices.items():
        q_target[:, target_indices[name]] = q_source[:, src_idx]
    return q_target


def compute_reduced_to_fixed(ik: Any, fixed_model: Any, fixed_data: Any) -> np.ndarray:
    """Rigid transform mapping reduced-model world poses into fixed-model world.

    FK a common link at the neutral configuration in both models:
    T_r2f = T_fixed_link @ inv(T_reduced_link). Should be approximately pure yaw.
    """
    import pinocchio as pin  # noqa: PLC0415

    fixed_frames = {str(frame.name) for frame in fixed_model.frames}
    reduced_frames = {str(frame.name) for frame in ik.robot.model.frames}
    common = next(
        (
            name
            for name in T_R2F_CANDIDATE_FRAMES
            if name in fixed_frames and name in reduced_frames
        ),
        None,
    )
    if common is None:
        raise ValueError(
            f"No common frame among {T_R2F_CANDIDATE_FRAMES} in both URDF models"
        )

    reduced_model = ik.robot.model
    reduced_data = ik.robot.data
    pin.forwardKinematics(reduced_model, reduced_data, pin.neutral(reduced_model))
    pin.updateFramePlacements(reduced_model, reduced_data)
    t_reduced = reduced_data.oMf[reduced_model.getFrameId(common)].homogeneous.copy()

    pin.forwardKinematics(fixed_model, fixed_data, pin.neutral(fixed_model))
    pin.updateFramePlacements(fixed_model, fixed_data)
    t_fixed = fixed_data.oMf[fixed_model.getFrameId(common)].homogeneous.copy()

    t_r2f = t_fixed @ np.linalg.inv(t_reduced)
    rotvec = Rotation.from_matrix(t_r2f[:3, :3]).as_rotvec()
    print(
        f"T_reduced_to_fixed (frame {common!r}): rotvec={np.array2string(rotvec, precision=4)} "
        f"(yaw + frozen-torso pitch; diagnostics only, never applied to world poses), "
        f"translation={np.array2string(t_r2f[:3, 3], precision=4)}"
    )
    return t_r2f


def fk_frames_on_model(
    model: Any,
    data: Any,
    q_fixed: np.ndarray,
    q_indices: dict[str, int],
    frame_names: list[str],
) -> np.ndarray:
    """FK world poses (T, len(frame_names), 7) of the named frames on the given model."""
    import pinocchio as pin  # noqa: PLC0415

    frame_ids = [model.getFrameId(name) for name in frame_names]
    num_frames = q_fixed.shape[0]
    out = np.zeros((num_frames, len(frame_ids), 7), dtype=np.float32)
    q_model = pin.neutral(model)
    for t in range(num_frames):
        q_model[:] = 0.0
        for idx in q_indices.values():
            q_model[idx] = q_fixed[t, idx]
        pin.forwardKinematics(model, data, q_model)
        pin.updateFramePlacements(model, data)
        for i, fid in enumerate(frame_ids):
            placement = data.oMf[fid]
            quat = pin.Quaternion(placement.rotation)
            out[t, i] = [*placement.translation, quat.w, quat.x, quat.y, quat.z]
    return out


def ee_poses_on_model(
    model: Any, data: Any, q_fixed: np.ndarray, q_indices: dict[str, int]
) -> np.ndarray:
    """FK world poses (T, 2, 7) of the two EE links on the given model."""
    return fk_frames_on_model(model, data, q_fixed, q_indices, EE_LINK_NAMES)


def resample_linear(
    values: np.ndarray, t_old: np.ndarray, t_new: np.ndarray
) -> np.ndarray:
    """Linearly resample a (T, ...) array along the time axis."""
    flat = values.reshape(values.shape[0], -1)
    resampled = np.stack(
        [np.interp(t_new, t_old, flat[:, j]) for j in range(flat.shape[1])], axis=1
    )
    return resampled.reshape(len(t_new), *values.shape[1:])


def resample_wxyz_slerp(
    wxyz: np.ndarray, t_old: np.ndarray, t_new: np.ndarray
) -> np.ndarray:
    """Slerp-resample a (T, 4) wxyz quaternion track along the time axis."""
    rotations = Rotation.from_quat(
        np.asarray(wxyz, dtype=np.float64), scalar_first=True
    )
    slerp = Slerp(t_old, rotations)
    return slerp(t_new).as_quat(scalar_first=True)


def resolve_existing_object_assets(
    output_root: Path, object_name: str
) -> tuple[Path, Path, float]:
    """Resolve an already-installed mesh/URDF pair instead of writing a scaled copy.

    Both embodiments of a clip describe the same physical object, so they should share
    one mesh. This reads the pair the loader/``generate_rigid_urdfs.py`` already
    installed and measures its radius, leaving the files untouched.

    Returns:
        Tuple of (mesh_path, urdf_path, mesh_radius).
    """
    mesh_path = output_root / f"{object_name}.obj"
    urdf_path = output_root / f"{object_name}{REUSED_URDF_SUFFIX}"
    missing = [str(p) for p in (mesh_path, urdf_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "--reuse_object_assets needs the object assets already installed in "
            f"--output_root; missing: {', '.join(missing)}. Run the loader / "
            "generate_rigid_urdfs.py for this sequence first, or drop the flag to write "
            "a private scaled copy."
        )
    vertices = [
        [float(v) for v in line.split()[1:4]]
        for line in mesh_path.read_text().splitlines()
        if line.startswith("v ")
    ]
    if not vertices:
        raise ValueError(f"No vertices parsed from {mesh_path}")
    mesh_radius = float(np.linalg.norm(np.asarray(vertices), axis=1).max())
    return mesh_path, urdf_path, mesh_radius


def write_scaled_object_assets(
    bundle_dir: Path,
    output_root: Path,
    object_name: str,
    vertex_scale: float,
    src_mesh_override: Path | None = None,
) -> tuple[Path, Path, float]:
    """Write the vertex-scaled mesh copy, its materials, and a minimal rigid URDF.

    Material/texture sidecars are copied under ``{object_name}_``-prefixed names (with
    the ``mtllib``/texture references rewritten) so that multiple objects sharing
    ``output_root`` do not overwrite each other's generic ``material.mtl`` /
    ``material_0.png`` exports. ``src_mesh_override`` uses an external OBJ (already metric,
    typically ``vertex_scale=1.0``) instead of the bundle reconstruction mesh — e.g. to
    substitute a cleaned-up scan of the real object.

    Returns:
        Tuple of (mesh_path, urdf_path, mesh_radius) where the radius is the max
        vertex distance from the object origin after scaling.
    """
    src_mesh = (
        src_mesh_override if src_mesh_override is not None else bundle_dir / "mesh.obj"
    )
    if not src_mesh.is_file():
        raise FileNotFoundError(f"Object mesh not found: {src_mesh}")
    output_root.mkdir(parents=True, exist_ok=True)

    mesh_path = output_root / f"{object_name}.obj"
    vertices: list[np.ndarray] = []
    with src_mesh.open() as fin, mesh_path.open("w") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                xyz = np.array([float(v) for v in parts[1:4]]) * vertex_scale
                rest = parts[4:]
                fout.write("v " + " ".join(f"{v:.8f}" for v in xyz))
                if rest:
                    fout.write(" " + " ".join(rest))
                fout.write("\n")
                vertices.append(xyz)
            elif line.startswith("mtllib "):
                fout.write(
                    f"mtllib {object_name}_{line.split(maxsplit=1)[1].strip()}\n"
                )
            else:
                fout.write(line)
    if not vertices:
        raise ValueError(f"No vertices parsed from {src_mesh}")
    verts = np.asarray(vertices)

    mesh_dir = src_mesh.parent
    textures = sorted(mesh_dir.glob("*.png")) + sorted(mesh_dir.glob("*.jpg"))
    for texture in textures:
        shutil.copy2(texture, output_root / f"{object_name}_{texture.name}")
    for mtl in sorted(mesh_dir.glob("*.mtl")):
        text = mtl.read_text()
        for texture in textures:
            text = text.replace(texture.name, f"{object_name}_{texture.name}")
        (output_root / f"{object_name}_{mtl.name}").write_text(text)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extents = np.maximum(bbox_max - bbox_min, 1e-4)
    center = 0.5 * (bbox_min + bbox_max)
    mass = OBJECT_MASS_KG
    ixx = mass / 12.0 * (extents[1] ** 2 + extents[2] ** 2)
    iyy = mass / 12.0 * (extents[0] ** 2 + extents[2] ** 2)
    izz = mass / 12.0 * (extents[0] ** 2 + extents[1] ** 2)
    mesh_radius = float(np.linalg.norm(verts, axis=1).max())

    urdf_path = output_root / f"{object_name}.urdf"
    urdf_path.write_text(
        f"""<?xml version="1.0"?>
<robot name="{object_name}">
  <link name="object">
    <inertial>
      <origin xyz="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.8f}" ixy="0" ixz="0" iyy="{iyy:.8f}" iyz="0" izz="{izz:.8f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_path.name}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_path.name}"/></geometry>
    </collision>
  </link>
</robot>
"""
    )
    return mesh_path, urdf_path, mesh_radius


def render_verification_video(
    ik: Any,
    q_reduced_out: np.ndarray,
    object_name: str,
    object_mesh_local: Path,
    object_position_out: np.ndarray,
    object_wxyz_out: np.ndarray,
    ee_pose_w: np.ndarray,
    fps: float,
    out_path: Path,
    camera_azimuth_deg: float = 225.0,
    camera_elevation_deg: float = 30.0,
    camera_padding: float = 1.5,
) -> None:
    """Render a headless MP4 of the retargeted robot + object (reduced world) for QA.

    Reuses ``OfflineVideoRenderer`` (pyrender/EGL) with the IK robot's own visual model,
    so it runs inside the retarget container without Isaac Sim. Best-effort: the caller
    guards this in try/except since it is verification-only.
    """
    import trimesh  # noqa: PLC0415
    from _offline_video import OfflineVideoRenderer  # noqa: PLC0415

    class _Kin:
        """Minimal adapter exposing ``.robot`` for OfflineVideoRenderer.add_robot."""

        def __init__(self, robot: Any) -> None:
            self.robot = robot

    renderer = OfflineVideoRenderer(fps=int(round(fps)))
    try:
        renderer.add_robot("vega", _Kin(ik.robot))
        renderer.add_object(
            object_name, trimesh.load(str(object_mesh_local), force="mesh")
        )
        pts = np.concatenate(
            [ee_pose_w[:, :, :3].reshape(-1, 3), object_position_out.reshape(-1, 3)],
            axis=0,
        )
        renderer.fit_camera(
            pts,
            elevation_deg=camera_elevation_deg,
            azimuth_deg=camera_azimuth_deg,
            padding=camera_padding,
        )
        rot = Rotation.from_quat(object_wxyz_out, scalar_first=True).as_matrix()
        for t in range(q_reduced_out.shape[0]):
            renderer.update_robot(
                "vega", np.asarray(q_reduced_out[t], dtype=np.float64)
            )
            transform = np.eye(4)
            transform[:3, :3] = rot[t]
            transform[:3, 3] = object_position_out[t]
            renderer.update_object(object_name, transform)
            renderer.capture()
        renderer.save(out_path)
    finally:
        renderer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse retarget-driver command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument(
        "--loaded_dir",
        type=Path,
        default=None,
        help="Loader-produced ManoSharpaData root. When set, use its world-space "
        "MANO joints/orientations and keep MANO FK inside the loader image.",
    )
    parser.add_argument(
        "--mano_model_dir",
        type=Path,
        default=None,
        help="Legacy direct-FK fallback; requires manotorch in the current environment.",
    )
    parser.add_argument("--sequence_id", default="tissue_box_simple")
    parser.add_argument("--object_name", default="tissue_box_simple")
    parser.add_argument(
        "--object_mesh",
        type=Path,
        default=None,
        help="External object OBJ to use instead of the bundle reconstruction mesh "
        "(already metric; applied at scale 1.0), e.g. a cleaned-up scan of the real object.",
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Where the verification video, IK sidecar and report go. Kept out of "
        "--output_root so regenerable QA output never enters the asset tree.",
    )
    parser.add_argument(
        "--reuse_object_assets",
        action="store_true",
        help="Point the parquet at the object mesh/URDF already in --output_root "
        f"(``<object_name>.obj`` + ``<object_name>{REUSED_URDF_SUFFIX}``, the loader's "
        "convention) instead of writing a scaled copy. Use this so the floating-hand and "
        "whole-body retargets of one clip share a single object asset.",
    )
    parser.add_argument(
        "--scene_scale",
        default="1.0",
        help="Uniform scale for world translations AND object mesh vertices. Defaults to "
        "1.0: the pipeline contract is that bundles are metric (see the onboard_ego_clip "
        "Step-0 hand-scale check), so no correction is needed. 'auto' = "
        "2/(mean(sL)+mean(sR)) rescales by the fitted hand scale -- an escape hatch for a "
        "known non-metric bundle, NOT a default. MANO betas absorb real anatomy at "
        "+/-10-15%%, so a hand scale near 1.1 is usually a large hand, not an oversized "
        "world; rescaling on that evidence silently shrinks the whole scene and desyncs "
        "the object mesh from the other embodiment's copy.",
    )
    parser.add_argument("--mano_to_robot_scale", type=float, default=1.2)
    parser.add_argument("--source_fps", type=float, default=30.0)
    parser.add_argument("--target_fps", type=float, default=50.0)
    parser.add_argument("--yaw_deg", type=float, default=180.0)
    parser.add_argument("--extra_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--first_frame_extra_solves", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=DEFAULT_CONTACT_THRESHOLD_M,
        help="Probe-to-surface distance (m) for hand-object contact activation.",
    )
    parser.add_argument(
        "--min_consecutive",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE,
        help="Minimum consecutive sub-threshold frames to keep a contact (de-flicker).",
    )
    parser.add_argument(
        "--no_contact_motion_gate",
        dest="contact_motion_gate",
        action="store_false",
        help="Disable suppressing contacts while the object rests on its support.",
    )
    parser.set_defaults(contact_motion_gate=True)
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render a headless MP4 (robot + object + active contacts) next to the parquet.",
    )
    parser.add_argument(
        "--video_azimuth_deg",
        type=float,
        default=225.0,
        help="Verification-video camera azimuth about world +Z. The eye sits at "
        "(cos az, sin az) from the subject, so 225 deg is a three-quarter view of the "
        "robot's FRONT; 45 deg is the same view of its back, which hides the grasp "
        "behind the torso.",
    )
    parser.add_argument(
        "--video_elevation_deg",
        type=float,
        default=30.0,
        help="Verification-video camera elevation above the horizontal.",
    )
    parser.add_argument(
        "--video_padding",
        type=float,
        default=1.5,
        help="Verification-video framing slack; >1 zooms out around the hands + object.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Retarget the bundle, print the JSON report, and optionally write outputs."""
    args = parse_args(argv)
    warnings: list[str] = []

    bundle_dir = args.bundle_dir.resolve()
    output_root = args.output_root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data = load_bundle(bundle_dir, args.frame_start, args.max_frames, warnings)
    num_frames = data["camera_to_world"].shape[0]
    frame_indices = np.arange(args.frame_start, args.frame_start + num_frames)

    mean_scale_left = float(data["left_scale"].mean())
    mean_scale_right = float(data["right_scale"].mean())
    if args.scene_scale == "auto":
        scene_scale = 2.0 / (mean_scale_left + mean_scale_right)
    else:
        scene_scale = float(args.scene_scale)
    print(
        f"hand scales: left={mean_scale_left:.4f}, right={mean_scale_right:.4f}; "
        f"scene_scale={scene_scale:.4f}"
    )

    loaded_data: Any | None = None
    loaded_object_position_w: np.ndarray | None = None
    loaded_object_rotation_w: np.ndarray | None = None
    joints_w: dict[str, np.ndarray] = {}
    rotations_w: dict[str, np.ndarray] = {}
    if args.loaded_dir is not None:
        from robotic_grounding.retarget.data_logger import (  # noqa: PLC0415
            ManoSharpaData,
        )

        loaded_data = ManoSharpaData.from_parquet(
            str(args.loaded_dir), filters=[("sequence_id", "=", args.sequence_id)]
        )
        loaded_slice = slice(args.frame_start, args.frame_start + num_frames)
        loaded_joints: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            joints = np.asarray(
                getattr(loaded_data, f"mano_{side}_joints"), dtype=np.float64
            )[loaded_slice]
            joints_wxyz = np.asarray(
                getattr(loaded_data, f"mano_{side}_joints_wxyz"), dtype=np.float64
            )[loaded_slice]
            if joints.shape != (num_frames, NUM_MANO_JOINTS, 3):
                raise ValueError(
                    f"Loaded {side} MANO joints have shape {joints.shape}; expected "
                    f"({num_frames}, {NUM_MANO_JOINTS}, 3)"
                )
            loaded_joints[side] = joints
            rotations_w[side] = (
                Rotation.from_quat(joints_wxyz.reshape(-1, 4), scalar_first=True)
                .as_matrix()
                .reshape(num_frames, NUM_MANO_JOINTS, 3, 3)
            )
        body_positions = np.asarray(loaded_data.object_body_position, dtype=np.float64)[
            loaded_slice
        ]
        body_wxyz = np.asarray(loaded_data.object_body_wxyz, dtype=np.float64)[
            loaded_slice
        ]
        if body_positions.shape[1] != 1 or body_wxyz.shape[1] != 1:
            raise ValueError("Vega ego reconstruction expects exactly one object body")
        loaded_object_rotation = Rotation.from_quat(
            body_wxyz[:, 0], scalar_first=True
        ).as_matrix()
        (
            joints_w,
            loaded_object_position_w,
            loaded_object_rotation_w,
        ) = reconcile_loaded_world_tracks(
            loaded_joints,
            body_positions[:, 0],
            loaded_object_rotation,
            data,
            scene_scale,
        )
        print(f"using loader-produced MANO world tracks from {args.loaded_dir}")
    else:
        if args.mano_model_dir is None:
            raise ValueError("pass --loaded_dir or --mano_model_dir")
        # Legacy direct MANO FK in camera frame (left params mirror-corrected first).
        import torch  # noqa: PLC0415
        from read_mano import MANO  # noqa: PLC0415

        device = torch.device(args.device)
        mano = MANO(
            mano_assets_root=str(args.mano_model_dir),
            gender="neutral",
            device=device,
            flat_hand_mean=True,
            center_idx=None,
        )
        left_orient, left_fingers = correct_left_hand_params(
            data["left_orient_cam"], data["left_finger_pose"]
        )
        orient_by_side = {"left": left_orient, "right": data["right_orient_cam"]}
        fingers_by_side = {"left": left_fingers, "right": data["right_finger_pose"]}
        for side in ("left", "right"):
            joints_cam, joints_wxyz_cam = mano_camera_fk(
                mano,
                side,
                data[f"{side}_betas"],
                orient_by_side[side],
                fingers_by_side[side],
                device,
            )
            joints_w[side], rotations_w[side] = hand_world_tracks(
                joints_cam,
                joints_wxyz_cam,
                data["camera_to_world"],
                data[f"{side}_wrist_to_world"],
                scene_scale,
            )

    # --- Object world poses (orthonormalized rotation, scaled translation) ---
    # The bundle's object 3x3 blocks may embed a uniform scale that SVD
    # orthonormalization strips from the pose; recover it and fold it (together with
    # the bundle's explicit object_scale, as in ego_recon_loader) into the mesh
    # vertex scale so the sim object stays consistent with its own trajectory.
    scale_per_frame = np.cbrt(np.linalg.det(data["object_to_world"][:, :3, :3]))
    if np.any(scale_per_frame <= 0.0):
        raise ValueError(
            "object_to_world rotation blocks have non-positive determinant"
        )
    embedded_scale = float(scale_per_frame.mean())
    spread = float(scale_per_frame.max() - scale_per_frame.min())
    if spread > EMBEDDED_SCALE_SPREAD_TOL:
        raise ValueError(
            f"object_to_world embedded scale varies across frames (spread {spread:.4g})"
        )
    if abs(embedded_scale - 1.0) > EMBEDDED_SCALE_WARN_TOL:
        warnings.append(
            f"object_to_world embeds a uniform scale {embedded_scale:.4f}; folding it "
            "into the object mesh vertex scale."
        )
        print(f"WARNING: {warnings[-1]}")
    object_scale = float(data["object_scale"])
    object_vertex_scale = scene_scale * embedded_scale * object_scale
    print(
        f"object scales: embedded={embedded_scale:.4f}, object_scale={object_scale:.4f}, "
        f"mesh vertex scale={object_vertex_scale:.4f}"
    )
    if loaded_data is None:
        object_rotation_w = orthonormalize_rotations(data["object_to_world"][:, :3, :3])
        object_position_w = scene_scale * data["object_to_world"][:, :3, 3]
    else:
        assert loaded_object_position_w is not None
        assert loaded_object_rotation_w is not None
        object_position_w = loaded_object_position_w
        object_rotation_w = loaded_object_rotation_w

    # --- Placement into the reduced-URDF robot world ---
    ik_module = load_ik_module()
    ik = ik_module.DexmateSharpaIK(max_iters=args.max_iters)
    r_place, t_place = compute_placement(
        ik,
        joints_w["left"][:, 0],
        joints_w["right"][:, 0],
        args.yaw_deg,
        np.asarray(args.extra_offset, dtype=np.float64),
    )
    t_place_matrix = np.eye(4, dtype=np.float64)
    t_place_matrix[:3, :3] = r_place
    t_place_matrix[:3, 3] = t_place

    placed_joints = {
        side: joints_w[side] @ r_place.T + t_place for side in ("left", "right")
    }
    placed_rotations = {
        side: np.einsum("ij,tkjl->tkil", r_place, rotations_w[side])
        for side in ("left", "right")
    }
    object_position_placed = object_position_w @ r_place.T + t_place
    object_rotation_placed = np.einsum("ij,tjk->tik", r_place, object_rotation_w)
    object_wxyz_placed = matrices_to_wxyz(object_rotation_placed)

    for side in ("left", "right"):
        wrist = placed_joints[side][:, 0]
        print(
            f"placed {side} wrist range: min={np.array2string(wrist.min(axis=0), precision=3)} "
            f"max={np.array2string(wrist.max(axis=0), precision=3)}"
        )

    # --- IK solve loop ---
    task_names = ik.task_frame_names
    num_tasks = len(task_names)
    orientation_active = np.array(
        [
            ik_module.DEFAULT_FRAME_TASK_SPECS[name].orientation_cost > 0.0
            for name in task_names
        ]
    )
    q_solutions = np.zeros((num_frames, ik.robot.model.nq), dtype=np.float64)
    converged = np.zeros(num_frames, dtype=bool)
    iterations = np.zeros(num_frames, dtype=np.int32)
    position_errors = np.zeros((num_frames, num_tasks), dtype=np.float64)
    orientation_errors = np.zeros((num_frames, num_tasks), dtype=np.float64)
    target_positions = np.zeros((num_frames, num_tasks, 3), dtype=np.float64)
    target_wxyz = np.zeros((num_frames, num_tasks, 4), dtype=np.float64)

    q_prev = ik.q_reference
    for t in range(num_frames):
        targets = build_frame_targets(
            ik_module,
            task_names,
            placed_joints,
            placed_rotations,
            t,
            args.mano_to_robot_scale,
        )
        result = ik.solve(targets, q_init=q_prev)
        if t == 0:
            for _ in range(args.first_frame_extra_solves):
                if result.converged:
                    break
                result = ik.solve(targets, q_init=result.q)
        q_prev = result.q
        q_solutions[t] = result.q
        converged[t] = result.converged
        iterations[t] = result.iterations
        for k, name in enumerate(task_names):
            error = np.asarray(result.frame_errors[name], dtype=np.float64)
            position_errors[t, k] = np.linalg.norm(error[:3])
            orientation_errors[t, k] = np.linalg.norm(error[3:])
            target_positions[t, k] = targets[name].position
            target_wxyz[t, k] = targets[name].wxyz
        if (t + 1) % 50 == 0 or t == num_frames - 1:
            print(
                f"solved {t + 1}/{num_frames} frames "
                f"(converged so far: {int(converged[: t + 1].sum())})"
            )

    # --- Report ---
    wrist_mask = np.array([name.endswith("hand_C_MC") for name in task_names])
    tip_mask = np.array([name.endswith("_fingertip") for name in task_names])
    mp_mask = ~(wrist_mask | tip_mask)

    report: dict[str, Any] = {
        "sequence_id": args.sequence_id,
        "bundle_dir": str(bundle_dir),
        "frames": int(num_frames),
        "frame_start": int(args.frame_start),
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "scene_scale": float(scene_scale),
        "object_embedded_scale": embedded_scale,
        "object_scale": object_scale,
        "object_vertex_scale": float(object_vertex_scale),
        "hand_scale_mean": {"left": mean_scale_left, "right": mean_scale_right},
        "mano_to_robot_scale": float(args.mano_to_robot_scale),
        "gravity_alignment_applied": True,
        "converged_frames": int(converged.sum()),
        "converged_pct": float(100.0 * converged.mean()),
        "iterations": {
            "mean": float(iterations.mean()),
            "p95": float(np.percentile(iterations, PERCENTILE_95)),
            "max": int(iterations.max()),
        },
        "position_error_m": {
            "wrists": error_stats(position_errors[:, wrist_mask], frame_indices),
            "fingertips": error_stats(position_errors[:, tip_mask], frame_indices),
            "mps": error_stats(position_errors[:, mp_mask], frame_indices),
        },
        "active_orientation_error_rad": error_stats(
            orientation_errors[:, orientation_active], frame_indices
        ),
        "placement": {
            "yaw_deg": float(args.yaw_deg),
            "extra_offset": [float(v) for v in args.extra_offset],
            "T_place": t_place_matrix.tolist(),
        },
        "warnings": warnings,
    }

    if not args.save:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    # --- Fixed-model remap ---
    fixed_model, fixed_data = build_fixed_model()
    reduced_indices = joint_q_indices(ik.robot.model)
    fixed_indices = joint_q_indices(fixed_model)
    fixed_joint_names = [str(name) for name in fixed_model.names[1:]]
    reduced_joint_names = list(ik.robot_joint_names)
    q_solutions = clamp_q_to_limits(q_solutions, ik.robot.model, JOINT_LIMIT_MARGIN_RAD)
    q_fixed = remap_q_by_name(q_solutions, reduced_indices, fixed_indices)
    q_fixed = clamp_q_to_limits(q_fixed, fixed_model, JOINT_LIMIT_MARGIN_RAD)
    t_r2f = compute_reduced_to_fixed(ik, fixed_model, fixed_data)
    report["T_reduced_to_fixed"] = t_r2f.tolist()

    # Object poses in the fixed-model world.
    object_pose_reduced = np.tile(np.eye(4), (num_frames, 1, 1))
    object_pose_reduced[:, :3, :3] = object_rotation_placed
    object_pose_reduced[:, :3, 3] = object_position_placed
    object_pose_fixed = np.einsum("ij,tjk->tik", t_r2f, object_pose_reduced)
    object_position_fixed = object_pose_fixed[:, :3, 3]
    object_wxyz_fixed = matrices_to_wxyz(object_pose_fixed[:, :3, :3])

    output_root.mkdir(parents=True, exist_ok=True)

    # --- Object assets ---
    # `--reuse_object_assets` names specific files the caller expects to exist, so it
    # resolves OUTSIDE the fallback below: a missing shared mesh must fail loudly rather
    # than silently degrade to an objectless parquet (no contacts, no force closure).
    object_fields: dict[str, Any] = {}
    object_mesh_local: Path | None = None
    reused: tuple[Path, Path, float] | None = None
    if args.reuse_object_assets:
        if args.object_mesh is not None:
            raise ValueError(
                "--reuse_object_assets and --object_mesh are mutually exclusive"
            )
        reused = resolve_existing_object_assets(output_root, args.object_name)
        if abs(object_vertex_scale - 1.0) > EMBEDDED_SCALE_WARN_TOL:
            warnings.append(
                f"reusing the installed object mesh, but this run's vertex scale is "
                f"{object_vertex_scale:.4f}; the shared mesh will not match this "
                "motion's world. Re-run with --scene_scale 1.0 on a metric bundle, "
                "or drop --reuse_object_assets."
            )
            print(f"WARNING: {warnings[-1]}")

    # Writing a private copy is best-effort: fall back to an objectless parquet.
    try:
        if reused is not None:
            mesh_path, urdf_path, mesh_radius = reused
        else:
            # An external mesh is already metric, so it bypasses the bundle-derived
            # vertex scale.
            vscale = 1.0 if args.object_mesh is not None else object_vertex_scale
            mesh_path, urdf_path, mesh_radius = write_scaled_object_assets(
                bundle_dir, output_root, args.object_name, vscale, args.object_mesh
            )
        object_mesh_local = mesh_path
        object_fields = {
            "object_name": args.object_name,
            "safe_object_name": args.object_name,
            "object_body_names": [args.object_name],
            "safe_object_body_names": [args.object_name],
            "object_mesh_paths": [to_container_path(mesh_path)],
            "object_urdf_paths": [to_container_path(urdf_path)],
            "object_mesh_radius": [mesh_radius],
        }
        verb = "reused" if args.reuse_object_assets else "wrote"
        print(
            f"{verb} object assets: {mesh_path} / {urdf_path} "
            f"(radius {mesh_radius:.3f} m)"
        )
    except (OSError, ValueError) as exc:
        warnings.append(f"object assets unavailable, writing objectless parquet: {exc}")
        print(f"WARNING: {warnings[-1]}")

    # --- Resample to target fps (needs >= 2 frames; Slerp rejects a single rotation) ---
    duration = (num_frames - 1) / args.source_fps
    if num_frames > 1 and abs(args.target_fps - args.source_fps) > RESAMPLE_FPS_EPSILON:
        t_old = np.arange(num_frames) / args.source_fps
        t_new = np.arange(0.0, duration + 0.5 / args.target_fps, 1.0 / args.target_fps)
        t_new = np.clip(t_new, 0.0, duration)
        q_out = resample_linear(q_fixed, t_old, t_new)
        q_reduced_out = resample_linear(q_solutions, t_old, t_new)
        object_position_out = resample_linear(object_position_placed, t_old, t_new)
        object_wxyz_out = resample_wxyz_slerp(object_wxyz_placed, t_old, t_new)
        print(
            f"resampled {args.source_fps:g} -> {args.target_fps:g} fps: "
            f"{num_frames} -> {len(t_new)} frames"
        )
    else:
        if (
            num_frames == 1
            and abs(args.target_fps - args.source_fps) > RESAMPLE_FPS_EPSILON
        ):
            warnings.append(
                "single-frame sequence: skipping resample, passing the frame through at "
                f"target_fps={args.target_fps:g}"
            )
            print(f"WARNING: {warnings[-1]}")
        q_out = q_fixed
        q_reduced_out = q_solutions
        object_position_out = object_position_placed
        object_wxyz_out = object_wxyz_placed
    num_out = q_out.shape[0]

    # EE poses in the reduced world so they share the parquet's world frame with the
    # object poses and the replay-spawned robot.
    ee_pose_w = ee_poses_on_model(
        ik.robot.model, ik.robot.data, q_reduced_out, reduced_indices
    )

    # Hand keypoint frames (fingertip *_DP links) in the reduced world — consumed by
    # `motion_hand_keypoints_gaussian_exp`, which matches these names against the robot's
    # own `.*_DP` bodies. Per-side (T, 5, 7).
    hand_frames_w = [
        fk_frames_on_model(
            ik.robot.model,
            ik.robot.data,
            q_reduced_out,
            reduced_indices,
            HAND_KEYPOINT_FRAMES[side],
        )
        for side in ("left", "right")
    ]
    hand_frame_names = [HAND_KEYPOINT_FRAMES["left"], HAND_KEYPOINT_FRAMES["right"]]

    # Per-side finger joint columns of the fixed-order output (drives the finger-tracking
    # metrics; reader reorders these into IsaacLab joint order by name).
    finger_cols = {
        side: [
            i
            for i, n in enumerate(fixed_joint_names)
            if n.lower().startswith(side) and "_arm_j" not in n.lower()
        ]
        for side in ("left", "right")
    }
    hand_finger_joint_names = [
        [fixed_joint_names[i] for i in finger_cols["left"]],
        [fixed_joint_names[i] for i in finger_cols["right"]],
    ]
    hand_finger_joints = [
        q_out[:, finger_cols["left"]].astype(np.float32),
        q_out[:, finger_cols["right"]].astype(np.float32),
    ]

    identity_wxyz = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (num_out, 1))
    zeros3 = np.zeros((num_out, 3), np.float32)
    zero_active = np.zeros((num_out,), np.float32)

    # Hand-object contact labels via FK-probe / mesh proximity (needs the object mesh).
    # `hand_contact_active` un-gates `force_closure_reward`; absent it, that term is 0.
    contact_fields: dict[str, Any] = {}
    hand_contact_active = [zero_active, zero_active]
    if object_mesh_local is not None:
        probe_names = CONTACT_PROBE_FRAMES["left"] + CONTACT_PROBE_FRAMES["right"]
        probes = fk_frames_on_model(
            ik.robot.model, ik.robot.data, q_reduced_out, reduced_indices, probe_names
        )
        gate = (
            contact_labels.manipulation_window(object_position_out)
            if args.contact_motion_gate
            else None
        )
        contacts = contact_labels.compute_contacts(
            object_mesh_local,
            probes[..., :3].astype(np.float64),
            object_position_out.astype(np.float64),
            object_wxyz_out.astype(np.float64),
            threshold=args.contact_threshold,
            min_consecutive=args.min_consecutive,
            motion_gate=gate,
        )
        contact_fields = {
            "hand_object_contact_positions": [
                contacts["positions"][0].astype(np.float32),
                contacts["positions"][1].astype(np.float32),
            ],
            "hand_object_contact_normals": [
                contacts["normals"][0].astype(np.float32),
                contacts["normals"][1].astype(np.float32),
            ],
            "hand_object_contact_part_ids": [
                contacts["part_ids"][0].astype(np.int32),
                contacts["part_ids"][1].astype(np.int32),
            ],
        }
        hand_contact_active = [
            (contacts["part_ids"][0] > 0).any(axis=-1).astype(np.float32),
            (contacts["part_ids"][1] > 0).any(axis=-1).astype(np.float32),
        ]
        active_frames = [int((a > 0).sum()) for a in hand_contact_active]
        print(
            f"contacts: threshold {args.contact_threshold:.3f} m, gate {gate}; "
            f"active frames left/right = {active_frames[0]}/{active_frames[1]} of {num_out}"
        )

    md = MotionData(
        sequence_id=args.sequence_id,
        robot_name=ROBOT_NAME,
        motion_kind=MOTION_KIND,
        fps=float(args.target_fps),
        coord_frame=COORD_FRAME,
        source_dataset=SOURCE_DATASET,
        raw_motion_file=str(bundle_dir / "result.npz"),
        robot_joint_names=fixed_joint_names,
        robot_joint_positions=q_out.astype(np.float32),
        robot_root_position=zeros3,
        robot_root_wxyz=identity_wxyz,
        ee_link_names=EE_LINK_NAMES,
        ee_pose_w=ee_pose_w,
        hand_sides=["left", "right"],
        hand_frame_names=hand_frame_names,
        hand_frames_w=hand_frames_w,
        hand_finger_joint_names=hand_finger_joint_names,
        hand_finger_joints=hand_finger_joints,
        hand_contact_active=hand_contact_active,
    )
    for key, value in contact_fields.items():
        setattr(md, key, value)
    if object_fields:
        md.object_name = object_fields["object_name"]
        md.safe_object_name = object_fields["safe_object_name"]
        md.object_body_names = object_fields["object_body_names"]
        md.safe_object_body_names = object_fields["safe_object_body_names"]
        md.object_mesh_paths = object_fields["object_mesh_paths"]
        md.object_urdf_paths = object_fields["object_urdf_paths"]
        md.object_mesh_radius = object_fields["object_mesh_radius"]
        md.object_articulation = np.zeros((num_out,), np.float32)
        # Single rigid body: root pose == body pose. Replay/viser rebuild the object
        # trajectory from root position + axis-angle, so the real rotation must go here.
        md.object_root_axis_angle = (
            Rotation.from_quat(object_wxyz_out, scalar_first=True)
            .as_rotvec()
            .astype(np.float32)
        )
        md.object_root_position = object_position_out.astype(np.float32)
        md.object_body_position = object_position_out.astype(np.float32)[:, None, :]
        md.object_body_wxyz = object_wxyz_out.astype(np.float32)[:, None, :]

    partition_dir = save_motion_parquet(
        md, root_path=str(output_root), file_name=PARQUET_FILE_NAME
    )
    print(f"wrote motion_v1 parquet -> {partition_dir}")
    report["parquet_partition_dir"] = str(partition_dir)
    report["parquet_world"] = (
        "reduced (gravity-aligned placement world of vega_sharpa_reduced.urdf, root at "
        "origin; matches the ROBOT_REGISTRY 'vega_sharpa' replay spawn)"
    )

    # --- Optional verification video (headless MP4 next to the parquet) ---
    if args.video and object_mesh_local is not None:
        video_path = artifact_dir / f"{args.sequence_id}_vega_sharpa.mp4"
        try:
            render_verification_video(
                ik,
                q_reduced_out,
                args.object_name,
                object_mesh_local,
                object_position_out,
                object_wxyz_out,
                ee_pose_w,
                args.target_fps,
                video_path,
                camera_azimuth_deg=args.video_azimuth_deg,
                camera_elevation_deg=args.video_elevation_deg,
                camera_padding=args.video_padding,
            )
            report["video_path"] = str(video_path)
            print(f"wrote verification video -> {video_path}")
        except (
            Exception
        ) as exc:  # noqa: BLE001 — verification-only, never block the parquet
            warnings.append(f"verification video failed: {exc}")
            print(f"WARNING: {warnings[-1]}")
    elif args.video:
        print("WARNING: --video requested but no object mesh available; skipping video")

    # --- Native-fps diagnostics sidecar ---
    sidecar_path = artifact_dir / f"{args.sequence_id}_vega_sharpa_ik.npz"
    np.savez_compressed(
        sidecar_path,
        q=q_solutions,
        reduced_joint_names=np.array(reduced_joint_names),
        fixed_joint_names=np.array(fixed_joint_names),
        q_fixed_order=q_fixed,
        converged=converged,
        iterations=iterations,
        position_errors=position_errors,
        orientation_errors=orientation_errors,
        task_names=np.array(list(task_names)),
        target_positions=target_positions,
        target_wxyz=target_wxyz,
        object_pos_reduced=object_position_placed,
        object_wxyz_reduced=object_wxyz_placed,
        object_pos_fixed=object_position_fixed,
        object_wxyz_fixed=object_wxyz_fixed,
        scene_scale=np.float64(scene_scale),
        T_place=t_place_matrix,
        T_reduced_to_fixed=t_r2f,
        bundle_dir=np.array(str(bundle_dir)),
        npz_path=np.array(str(bundle_dir / "result.npz")),
        args_json=np.array(json.dumps({k: str(v) for k, v in vars(args).items()})),
    )
    print(f"wrote diagnostics sidecar -> {sidecar_path}")
    report["sidecar_path"] = str(sidecar_path)

    report_path = (
        args.report_path
        if args.report_path is not None
        else artifact_dir / f"{args.sequence_id}_vega_sharpa_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote report -> {report_path}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
