# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre- and post-planning validation for the G1 whole-body planner.

`warn_reference_issues` is called before planning starts. It checks reference-
owned data (input motion + on-disk assets) and prints warnings; it never
modifies anything. Issues here are the responsibility of the upstream data
producer (retargeting / asset pipeline).

`assert_motion_parquet_invariants` is called after the planner writes its
output parquet and hard-fails if any planner-owned invariant is violated, so
silent data corruption can't leak into training.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
# Pre-plan: reference-owned checks (warnings only)
# ----------------------------------------------------------------------------

REQUIRED_MOTION_FIELDS: tuple[str, ...] = (
    "object_body_position",
    "object_body_wxyz",
    "robot_left_wrist_position",
    "robot_left_wrist_wxyz",
    "robot_right_wrist_position",
    "robot_right_wrist_wxyz",
    "robot_left_frames",
    "robot_right_frames",
    "object_urdf_paths",
    "object_mesh_paths",
)

# Source FPS values seen across the retargeting pipelines we currently support.
# A non-listed value is a warning, not a hard failure.
EXPECTED_SOURCE_FPS: tuple[float, ...] = (30.0, 60.0, 100.0, 150.0)

# Optional fields — present-but-empty is fine, only warn if the user expects
# contact-aware downstream training.
OPTIONAL_CONTACT_FIELDS: tuple[str, ...] = (
    "mano_left_object_contact_positions",
    "mano_right_object_contact_positions",
    "mano_left_link_contact_positions",
    "mano_right_link_contact_positions",
)


def _has_nonempty(motion: Any, attr: str) -> bool:
    value = getattr(motion, attr, None)
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return value is not None


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.exists() else None


def walk_urdf_mesh_deps(urdf_path: Path) -> list[Path]:
    """Return absolute paths of every ``<mesh filename=...>`` referenced by URDF."""
    try:
        text = urdf_path.read_text()
    except OSError:
        return []
    deps: list[Path] = []
    for raw in re.findall(r'mesh\s+filename="([^"]+)"', text):
        if raw.startswith("package://"):
            deps.append(Path(raw))
        elif Path(raw).is_absolute():
            deps.append(Path(raw))
        else:
            deps.append((urdf_path.parent / raw).resolve())
    return deps


def warn_missing_urdf_mesh_deps(urdf_paths: list[str]) -> None:
    """Print one warning per missing visual/collision file referenced by URDFs.

    A URDF can resolve locally while its ``<mesh>`` children (visual STL,
    collision OBJ) aren't present — the IsaacLab importer would then error at
    scene-spawn time. Surfacing the gap now lets the user fix the workspace
    before they hit the import crash.
    """
    for urdf in urdf_paths:
        urdf_p = Path(urdf)
        if not urdf_p.exists():
            continue
        for dep in walk_urdf_mesh_deps(urdf_p):
            if not dep.exists():
                print(
                    f"  WARNING: URDF {urdf_p.name} references missing mesh {dep}; "
                    "copy this file into the workspace before training."
                )


def warn_reference_issues(motion: Any, ref_data: dict, robot_type: str) -> list[str]:
    """Print warnings for reference-owned problems before planning.

    Returns the list of warning lines emitted, for downstream logging / tests.
    Never raises — these are the upstream data producer's responsibility, and
    the planner can still produce output (it just may be incomplete).
    """
    warnings: list[str] = []

    def _warn(msg: str) -> None:
        line = f"  REFERENCE WARNING: {msg}"
        print(line)
        warnings.append(line)

    # 1. Required input fields
    missing = [f for f in REQUIRED_MOTION_FIELDS if not _has_nonempty(motion, f)]
    if missing:
        _warn(
            f"reference motion is missing required fields {missing}; "
            "planner output will have empty / zero values for the corresponding "
            "training-side columns."
        )

    # 2. Source FPS sanity
    src_fps = float(getattr(motion, "fps", 0.0) or 0.0)
    if src_fps <= 0.0:
        _warn(
            "reference motion has fps <= 0; downstream upsampling cannot infer source rate."
        )
    elif src_fps not in EXPECTED_SOURCE_FPS:
        _warn(
            f"reference motion fps={src_fps} is not in the expected set "
            f"{EXPECTED_SOURCE_FPS}; verify the contact upsample assumes the right rate."
        )

    # 3. Asset paths resolve
    for attr in ("object_urdf_paths", "object_mesh_paths"):
        paths = getattr(motion, attr, None) or []
        for p in paths:
            if _resolve_path(p) is None:
                _warn(
                    f"{attr} entry {p!r} does not resolve to an existing file; "
                    "scene spawn will fail unless the planner's path remapper recovers it."
                )

    # 4. URDF mesh dependencies
    for urdf in getattr(motion, "object_urdf_paths", None) or []:
        urdf_p = _resolve_path(urdf)
        if urdf_p is None:
            continue
        for dep in walk_urdf_mesh_deps(urdf_p):
            if not dep.exists():
                _warn(
                    f"URDF {urdf_p.name} references missing mesh {dep}; "
                    "copy this visual/collision file into the workspace before training."
                )

    # 5. Contact-aware downstream training will read these — empty == zero reward signal.
    empty_contacts = [
        f for f in OPTIONAL_CONTACT_FIELDS if not _has_nonempty(motion, f)
    ]
    if len(empty_contacts) == len(OPTIONAL_CONTACT_FIELDS):
        _warn(
            "reference motion has no MANO contact arrays on either side; "
            "any contact-conditioned reward / observation will be all-zero."
        )
    elif empty_contacts:
        _warn(f"reference motion is missing contact arrays {empty_contacts}.")

    # 6. ee_link_names label on the reference (informational; planner overrides this).
    ref_ee = getattr(motion, "ee_link_names", None)
    if ref_ee:
        expected = (
            ("left_hand_palm_link", "right_hand_palm_link")
            if robot_type == "dex3"
            else ("left_wrist_yaw_link", "right_wrist_yaw_link")
        )
        if tuple(ref_ee) != expected:
            _warn(
                f"reference ee_link_names={list(ref_ee)} disagrees with the planner's "
                f"convention {list(expected)} for robot_type={robot_type!r}; "
                "the planner will overwrite this label in the output parquet."
            )

    return warnings


# ----------------------------------------------------------------------------
# Post-plan: planner-owned invariants (hard failures)
# ----------------------------------------------------------------------------


EXPECTED_EE_LINK_NAMES: dict[str, list[str]] = {
    "sharpa": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    "dex3": ["left_hand_palm_link", "right_hand_palm_link"],
}

# Per-robot fingertip body suffixes that the training env expects to find in
# `robot_joint_names` (so `tracking_command._load_and_process_motion` can
# resolve `cfg.joint_names`). These are the joint names, not body names.
EXPECTED_FINGER_JOINT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "dex3": (re.compile(r"^(left|right)_hand_(thumb|index|middle)_\d_joint$"),),
    # Sharpa has 22+22 finger joints; the planner currently doesn't gate on
    # name patterns for sharpa, so leave the check open.
    "sharpa": (),
}

# Palm-vs-ee distance after the planner's V2P→planner transform. For dex3 the
# palm IS the EE (free-flyer URDF root), so they should coincide to numerical
# precision. For sharpa the EE (wrist_yaw_link) and palm (hand_C_MC) differ by
# the URDF fixed-joint offset (~4 cm); we don't check this side strictly.
PALM_EE_TOLERANCE_M: dict[str, float] = {
    "dex3": 0.01,
    "sharpa": 0.10,
}


class ParquetInvariantError(AssertionError):
    """Raised when the output parquet violates a planner-owned invariant."""


def _arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _load_md(parquet_path: Path) -> dict:
    """Load the parquet as a plain dict (one entry per Hive row)."""
    table = pq.read_table(str(parquet_path))
    data = table.to_pydict()
    # Each top-level column is a 1-row list; unwrap.
    return {k: v[0] for k, v in data.items()}


def _check_object_root(md: dict) -> list[str]:
    errs: list[str] = []
    bp = _arr(md.get("object_body_position", []))
    rp = md.get("object_root_position")
    if bp.ndim != 3:
        return [f"object_body_position has wrong rank {bp.shape} (expected (T,B,3))"]
    if rp is None:
        return ["object_root_position is None — required for articulated scene init"]
    rp_a = _arr(rp)
    if rp_a.shape[0] != bp.shape[0]:
        errs.append(
            f"object_root_position length {rp_a.shape[0]} != object_body_position length {bp.shape[0]}; "
            "they must share a frame count so the scene init pose matches the trajectory start."
        )
    if rp_a.shape[0] == bp.shape[0]:
        max_d = float(np.max(np.linalg.norm(rp_a - bp[:, 0, :], axis=-1)))
        if max_d > 1e-3:
            errs.append(
                f"object_root_position diverges from object_body_position[:,0] by {max_d:.4f} m; "
                "root must mirror body 0 so the env's reset pose lands where the trajectory starts."
            )

    raa = md.get("object_root_axis_angle")
    if raa is None:
        errs.append("object_root_axis_angle is None")
    else:
        raa_a = _arr(raa)
        if raa_a.shape[0] != bp.shape[0]:
            errs.append(
                f"object_root_axis_angle length {raa_a.shape[0]} != object_body_position length {bp.shape[0]}"
            )
    return errs


def _check_ee_link_names(md: dict, robot_type: str) -> list[str]:
    expected = EXPECTED_EE_LINK_NAMES.get(robot_type)
    if expected is None:
        return [f"unknown robot_type={robot_type!r}; cannot validate ee_link_names"]
    actual = list(md.get("ee_link_names") or [])
    if actual != expected:
        return [
            f"ee_link_names={actual} but expected {expected} for robot_type={robot_type!r}; "
            "this label tells the env which body the EE pose was recorded from, so a "
            "mismatch puts the reward target on the wrong link."
        ]
    return []


def _check_hand_frames_transform(md: dict, robot_type: str) -> list[str]:
    """Verify hand_frames_w[palm] coincides with ee_pose_w.

    For dex3 the palm IS the EE (free-flyer root); they should coincide to
    numerical precision. A larger gap means the planner-frame rigid transform
    was applied to one of these arrays but not the other.
    """
    errs: list[str] = []
    hfw = md.get("hand_frames_w")
    hfn = md.get("hand_frame_names")
    ee = md.get("ee_pose_w")
    sides = md.get("hand_sides") or []
    if hfw is None or not hfw or ee is None:
        return errs  # nothing to check
    hfw_a = _arr(hfw)  # (S, T, K, 7)
    ee_a = _arr(ee)  # (T, 2, 7)
    if hfw_a.ndim != 4 or ee_a.ndim != 3:
        return errs
    palm_name_by_robot = {
        "dex3": ("left_hand_palm_link", "right_hand_palm_link"),
        "sharpa": ("left_hand_C_MC", "right_hand_C_MC"),
    }
    palm_names = palm_name_by_robot.get(robot_type, ())
    tol = PALM_EE_TOLERANCE_M.get(robot_type, 0.10)
    for s_idx, side in enumerate(sides):
        if s_idx >= hfw_a.shape[0] or s_idx >= len(palm_names):
            continue
        names = list(hfn[s_idx]) if hfn and s_idx < len(hfn) else []
        target = palm_names[s_idx]
        if target not in names:
            continue
        k = names.index(target)
        palm_xyz = hfw_a[s_idx, :, k, :3]
        side_xyz = ee_a[:, s_idx, :3] if ee_a.shape[1] > s_idx else None
        if side_xyz is None or palm_xyz.shape[0] != side_xyz.shape[0]:
            continue
        max_d = float(np.max(np.linalg.norm(palm_xyz - side_xyz, axis=-1)))
        if max_d > tol:
            errs.append(
                f"hand_frames_w[{side}, palm] vs ee_pose_w[{side}] max distance "
                f"{max_d:.4f} m exceeds {tol} m; both columns describe the same body "
                "so they should agree — one of them is in the wrong frame."
            )
    return errs


def _check_fingers_in_joint_names(md: dict, robot_type: str) -> list[str]:
    patterns = EXPECTED_FINGER_JOINT_PATTERNS.get(robot_type) or ()
    if not patterns:
        return []
    names = list(md.get("robot_joint_names") or [])
    has_finger = any(p.match(n) for p in patterns for n in names)
    if not has_finger:
        return [
            "robot_joint_names has no finger joints; tracking_command resolves "
            "cfg.joint_names against this list and raises if a tracked joint is missing."
        ]
    return []


def _check_contact_field_shapes(md: dict) -> list[str]:
    """Confirm contact arrays have a sane (S, T, N, 3) shape and the right T."""
    errs: list[str] = []
    bp = _arr(md.get("object_body_position", []))
    if bp.ndim != 3:
        return errs
    T = bp.shape[0]
    for field in (
        "hand_link_contact_positions",
        "hand_link_contact_normals",
        "hand_object_contact_positions",
        "hand_object_contact_normals",
    ):
        value = md.get(field)
        if value is None:
            continue
        for s_idx, side_arr in enumerate(value):
            if side_arr is None or len(side_arr) == 0:
                continue
            side_a = _arr(side_arr)
            if side_a.ndim != 3:
                errs.append(f"{field}[side={s_idx}] has wrong rank {side_a.shape}")
                continue
            if side_a.shape[0] != T:
                errs.append(
                    f"{field}[side={s_idx}] frame count {side_a.shape[0]} != T={T}; "
                    "every per-frame contact array must match the body trajectory length."
                )
            if side_a.shape[-1] != 3:
                errs.append(
                    f"{field}[side={s_idx}] last dim {side_a.shape[-1]} != 3; "
                    "the part-id column is stored separately, only xyz should remain here."
                )

    # hand_contact_active must be per-side, length T, binary-ish.
    hca = md.get("hand_contact_active")
    if hca is not None:
        hca_a = _arr(hca)
        if hca_a.ndim == 2 and hca_a.shape[1] != T:
            errs.append(f"hand_contact_active shape {hca_a.shape} second dim != T={T}")
    return errs


def _check_asset_paths(md: dict) -> list[str]:
    errs: list[str] = []
    for field in ("object_urdf_paths", "object_mesh_paths"):
        for p in md.get(field) or []:
            if not p or not Path(p).exists():
                errs.append(
                    f"{field} entry {p!r} does not exist on disk; "
                    "the parquet must be self-contained against the current workspace."
                )
    # Walk URDF deps; missing visuals/collisions block scene spawn.
    for urdf in md.get("object_urdf_paths") or []:
        urdf_p = Path(urdf)
        if not urdf_p.exists():
            continue
        for dep in walk_urdf_mesh_deps(urdf_p):
            if not dep.exists():
                errs.append(
                    f"URDF {urdf_p.name} references missing mesh {dep}; "
                    "every visual/collision file the URDF lists must be present."
                )
    return errs


def assert_motion_parquet_invariants(
    parquet_path: str | Path,
    robot_type: str,
) -> None:
    """Hard-fail if the planner output violates any required invariant.

    Call this immediately after `save_motion_parquet` returns. Each check
    targets a downstream consumer contract; the error messages spell out the
    contract so the fix is obvious.
    """
    parquet_path = Path(parquet_path)
    # `parquet_path` is the partition directory returned by `save_motion_parquet`;
    # the actual parquet sits inside as `<sequence_id>.parquet` (or `data.parquet`).
    if parquet_path.is_dir():
        candidates = sorted(parquet_path.glob("*.parquet"))
        if not candidates:
            raise ParquetInvariantError(
                f"No .parquet file under {parquet_path} — cannot validate planner output."
            )
        parquet_file = candidates[0]
    else:
        parquet_file = parquet_path

    md = _load_md(parquet_file)
    errors: list[str] = []
    errors.extend(_check_object_root(md))
    errors.extend(_check_ee_link_names(md, robot_type))
    errors.extend(_check_hand_frames_transform(md, robot_type))
    errors.extend(_check_fingers_in_joint_names(md, robot_type))
    errors.extend(_check_contact_field_shapes(md))
    errors.extend(_check_asset_paths(md))

    if errors:
        header = f"Planner output {parquet_file} failed {len(errors)} invariant(s):"
        raise ParquetInvariantError("\n  - ".join([header, *errors]))

    print(f"  planner_validation: {parquet_file.name} passed all invariants.")
