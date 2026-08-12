#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the fully synthetic ``synthbox`` E2E fixtures from scratch.

``synthbox`` is a made-up, license-free dataset that replaces TACO in the
end-to-end tests. It contains a single sequence — two hands opening an
articulated box (two box primitives joined by a revolute hinge). Every motion
trajectory here is fabricated procedurally; nothing is derived from any
third-party dataset. The only strings reused are the Sharpa-Wave robot and MANO
*structural* link/joint names (they describe the robot/hand model, not motion),
embedded below so this generator has no dataset dependency.

It writes, under
``source/robotic_grounding/robotic_grounding/assets/``:

  human_motion_data/synthbox/
    synthbox_loaded/     — ManoSharpaData with mano_* + object_* filled, robot_*
                           empty (input to the retarget pipeline E2E test).
    synthbox_processed/  — same schema with robot_* filled by fabricated IK-like
                           values (input to the train E2E test).
    reconstructed_stage/<seq>_support.usda — hand-written support disk.
    object_assets/{meshes,urdfs}/synthbox/box/…  — box meshes (the URDF is
                           hand-authored and committed separately).
  meshes/synthbox/box/{bottom,top}.obj — mesh copy at the canonical asset path
                           that support-surface reconstruction reads by default.

Re-run after a schema change:  python scripts/make_synthbox_fixtures.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO_ROOT / "source" / "robotic_grounding" / "robotic_grounding" / "assets"
HMD = ASSET_DIR / "human_motion_data"
SYNTHBOX_DIR = HMD / "synthbox"
OBJECT_ASSETS = SYNTHBOX_DIR / "object_assets"
LOADED_DIR = SYNTHBOX_DIR / "synthbox_loaded"
PROCESSED_DIR = SYNTHBOX_DIR / "synthbox_processed"
RECON_DIR = SYNTHBOX_DIR / "reconstructed_stage"

SEQUENCE_ID = "synthbox_box_open_000"
ROBOT_NAME = "sharpa_wave"
FPS = 30.0
T = 155  # frame count (kept the same as the retired taco fixture)


def _load_manosharpadata() -> type:
    """Import ``ManoSharpaData`` from source without triggering package init."""
    path = ASSET_DIR.parent / "retarget" / "data_logger.py"
    spec = importlib.util.spec_from_file_location("_synthbox_data_logger", str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ManoSharpaData


# --------------------------------------------------------------------------- #
# Structural names (Sharpa-Wave robot + MANO hand model — not motion data).
# The V2D command reorders finger joints / indexes frames *by name*, so these
# must be the real model names; the numeric trajectories are all fabricated.
# --------------------------------------------------------------------------- #
def _finger_joint_names(side: str) -> list[str]:
    stems = [
        "thumb_CMC_FE",
        "thumb_CMC_AA",
        "thumb_MCP_FE",
        "thumb_MCP_AA",
        "thumb_IP",
        "index_MCP_FE",
        "index_MCP_AA",
        "index_PIP",
        "index_DIP",
        "middle_MCP_FE",
        "middle_MCP_AA",
        "middle_PIP",
        "middle_DIP",
        "ring_MCP_FE",
        "ring_MCP_AA",
        "ring_PIP",
        "ring_DIP",
        "pinky_CMC",
        "pinky_MCP_FE",
        "pinky_MCP_AA",
        "pinky_PIP",
        "pinky_DIP",
    ]  # 22
    return [f"{side}_{s}" for s in stems]


def _frame_names(side: str) -> list[str]:
    stems = [
        "universe",
        "root_joint",
        "hand_C_MC",
        "thumb_CMC_FE",
        "thumb_CMC_VL",
        "thumb_CMC_VL_site",
        "thumb_CMC_AA",
        "thumb_MC",
        "thumb_MCP_FE",
        "thumb_MCP_VL",
        "thumb_MCP_VL_site",
        "thumb_MCP_AA",
        "thumb_PP",
        "thumb_IP",
        "thumb_DP",
        "thumb_DP_site",
        "thumb_tip_site",
        "index_MCP_FE",
        "index_MCP_VL",
        "index_MCP_VL_site",
        "index_MCP_AA",
        "index_PP",
        "index_PIP",
        "index_MP",
        "index_MP_site",
        "index_DIP",
        "index_DP",
        "index_DP_site",
        "index_tip_site",
        "middle_MCP_FE",
        "middle_MCP_VL",
        "middle_MCP_VL_site",
        "middle_MCP_AA",
        "middle_PP",
        "middle_PIP",
        "middle_MP",
        "middle_MP_site",
        "middle_DIP",
        "middle_DP",
        "middle_DP_site",
        "middle_tip_site",
        "ring_MCP_FE",
        "ring_MCP_VL",
        "ring_MCP_VL_site",
        "ring_MCP_AA",
        "ring_PP",
        "ring_PIP",
        "ring_MP",
        "ring_MP_site",
        "ring_DIP",
        "ring_DP",
        "ring_DP_site",
        "ring_tip_site",
        "pinky_CMC",
        "pinky_MC",
        "pinky_MC_site",
        "pinky_MCP_FE",
        "pinky_MCP_VL",
        "pinky_MCP_AA",
        "pinky_PP",
        "pinky_PIP",
        "pinky_MP",
        "pinky_MP_site",
        "pinky_DIP",
        "pinky_DP",
        "pinky_DP_site",
        "pinky_tip_site",
    ]  # 67
    # "universe" / "root_joint" are global frames without a side prefix.
    return [s if s in ("universe", "root_joint") else f"{side}_{s}" for s in stems]


def _frame_task_names(side: str) -> list[str]:
    stems = [
        "hand_C_MC",
        "thumb_MCP_VL_site",
        "thumb_tip_site",
        "index_MP_site",
        "index_tip_site",
        "middle_MP_site",
        "middle_tip_site",
        "ring_MP_site",
        "ring_tip_site",
        "pinky_MP_site",
        "pinky_tip_site",
    ]  # 11
    return [f"{side}_{s}" for s in stems]


MANO_LINK_NAMES = [
    "link_palm",
    "link_thumb1",
    "link_thumb2",
    "link_thumb3",
    "link_index1",
    "link_index2",
    "link_index3",
    "link_middle1",
    "link_middle2",
    "link_middle3",
    "link_ring1",
    "link_ring2",
    "link_ring3",
    "link_pinky1",
    "link_pinky2",
    "link_pinky3",
]  # 16

# --------------------------------------------------------------------------- #
# Object geometry (must match object_assets/urdfs/synthbox/box.urdf)
# --------------------------------------------------------------------------- #
BOTTOM_SIZE = np.array([0.20, 0.15, 0.10])  # base box (m)
TOP_SIZE = np.array([0.20, 0.15, 0.02])  # lid (m)
BOTTOM_CENTER = np.array([0.0, 0.0, 0.355])  # base rests on the support surface
HINGE = BOTTOM_CENTER + np.array([0.0, -0.075, 0.05])  # back-top edge of the base
LID_CLOSED_CENTER = np.array([0.0, 0.0, 0.415])  # lid center when angle == 0
OPEN_ANGLE = 1.6  # radians the lid opens to


def _box_obj(size: np.ndarray) -> str:
    """Return Wavefront OBJ text for an axis-aligned box centered at origin."""
    hx, hy, hz = (size / 2.0).tolist()
    verts = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 8, 7),
        (5, 7, 6),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]
    lines = ["# Synthetic axis-aligned box (Apache-2.0, NVIDIA)"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def _articulation_curve() -> np.ndarray:
    """Lid angle per frame: closed, then a smooth open, then held open."""
    theta = np.zeros(T)
    hold, ramp = 40, 70  # closed frames, then ramp frames
    s = np.linspace(0.0, 1.0, ramp)
    smooth = s * s * (3.0 - 2.0 * s)  # smoothstep
    theta[hold : hold + ramp] = OPEN_ANGLE * smooth
    theta[hold + ramp :] = OPEN_ANGLE
    return theta


def _object_trajectories() -> dict:
    """Fabricate per-frame object body poses for the articulated box."""
    theta = _articulation_curve()

    bottom_pos = np.tile(BOTTOM_CENTER, (T, 1))
    bottom_quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))  # wxyz identity

    top_pos = np.zeros((T, 3))
    top_quat = np.zeros((T, 4))
    offset = LID_CLOSED_CENTER - HINGE
    for t in range(T):
        rot = R.from_rotvec([theta[t], 0.0, 0.0])  # hinge about +x
        top_pos[t] = HINGE + rot.apply(offset)
        x, y, z, w = rot.as_quat()  # scipy returns xyzw
        top_quat[t] = [w, x, y, z]  # store wxyz

    body_position = np.stack([bottom_pos, top_pos], axis=1)  # (T, 2, 3)
    body_wxyz = np.stack([bottom_quat, top_quat], axis=1)  # (T, 2, 4)
    return {
        "object_articulation": theta.astype(np.float32).tolist(),
        "object_root_position": bottom_pos.astype(np.float32).tolist(),
        "object_root_axis_angle": np.zeros((T, 3), np.float32).tolist(),
        "object_body_position": body_position.astype(np.float32).tolist(),
        "object_body_wxyz": body_wxyz.astype(np.float32).tolist(),
    }


def _hand_template() -> np.ndarray:
    """A crude 21-joint open-hand skeleton in the wrist-local frame (meters)."""
    joints = [np.array([0.0, 0.0, 0.0])]  # wrist
    for finger in range(5):
        base_y = (finger - 2) * 0.02
        for seg in range(4):
            joints.append(np.array([0.02 * (seg + 1), base_y, 0.0]))
    return np.stack(joints, axis=0)  # (21, 3)


def _hand_trajectories(side: str) -> dict:
    """Fabricate per-frame MANO joints/quaternions for one hand near the box."""
    theta = _articulation_curve()
    s = theta / OPEN_ANGLE  # 0 -> 1 as the lid opens
    sign = 1.0 if side == "right" else -1.0
    template = _hand_template()

    joints = np.zeros((T, 21, 3))
    trans = np.zeros((T, 3))
    for t in range(T):
        wrist = np.array([sign * 0.18, -0.02 + 0.04 * s[t], 0.48 + 0.08 * s[t]])
        joints[t] = template + wrist
        trans[t] = wrist

    quats = np.tile([1.0, 0.0, 0.0, 0.0], (T, 21, 1))  # identity wxyz per joint
    return {
        f"mano_{side}_trans": trans.astype(np.float32).tolist(),
        f"mano_{side}_global_orient": np.zeros((T, 3), np.float32).tolist(),
        f"mano_{side}_finger_pose": np.zeros((T, 45), np.float32).tolist(),
        f"mano_{side}_joints": joints.astype(np.float32).tolist(),
        f"mano_{side}_joints_wxyz": quats.astype(np.float32).tolist(),
        f"mano_{side}_fitting_err": np.zeros(T, np.float32).tolist(),
        f"mano_{side}_tips_distance": np.full((T, 5), 0.1, np.float32).tolist(),
        f"mano_{side}_link_contact_positions": np.zeros(
            (T, 16, 3), np.float32
        ).tolist(),
        f"mano_{side}_link_contact_normals": np.zeros((T, 16, 3), np.float32).tolist(),
        f"mano_{side}_object_contact_positions": np.zeros(
            (T, 16, 3), np.float32
        ).tolist(),
        f"mano_{side}_object_contact_normals": np.zeros(
            (T, 16, 3), np.float32
        ).tolist(),
        f"mano_{side}_object_contact_part_ids": np.zeros((T, 16), np.int32).tolist(),
    }


def _robot_trajectories(side: str) -> dict:
    """Fabricate per-frame Sharpa robot targets (finite, normalized quats).

    These stand in for the retarget IK output so the train E2E test can drive a
    few iterations; they are not physically consistent, only schema-valid.
    """
    theta = _articulation_curve()
    s = theta / OPEN_ANGLE
    sign = 1.0 if side == "right" else -1.0

    wrist_pos = np.zeros((T, 3))
    for t in range(T):
        wrist_pos[t] = [sign * 0.18, -0.02 + 0.04 * s[t], 0.48 + 0.08 * s[t]]
    wrist_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))

    # 22 finger joints: gentle bounded oscillation.
    phase = np.linspace(0, 2 * np.pi, 22)
    finger_joints = 0.1 * np.sin(phase[None, :] + s[:, None] * np.pi)  # (T, 22)

    # 67 frames of 7-vec pose [x,y,z,qw,qx,qy,qz]; cluster around the wrist.
    frame_offsets = np.linspace(0.0, 0.08, 67)[None, :, None] * np.array(
        [1.0, 0.0, 0.0]
    )
    frames = np.zeros((T, 67, 7))
    frames[:, :, :3] = wrist_pos[:, None, :] + frame_offsets
    frames[:, :, 3] = 1.0  # identity quaternion (qw=1)
    return {
        f"robot_{side}_wrist_position": wrist_pos.astype(np.float32).tolist(),
        f"robot_{side}_wrist_wxyz": wrist_wxyz.astype(np.float32).tolist(),
        f"robot_{side}_finger_joints": finger_joints.astype(np.float32).tolist(),
        f"robot_{side}_frames": frames.astype(np.float32).tolist(),
        f"robot_{side}_frame_task_errors": np.zeros((T, 11), np.float32).tolist(),
        f"robot_{side}_num_optimization_iterations": np.ones(T, np.int32).tolist(),
    }


def _base_record() -> dict:
    """Metadata + object fields shared by the loaded and processed fixtures."""
    mesh_paths = [
        "object_assets/meshes/synthbox/box/bottom.obj",
        "object_assets/meshes/synthbox/box/top.obj",
    ]
    record: dict = {
        "sequence_id": SEQUENCE_ID,
        "raw_motion_file": "synthetic://synthbox/box_open_000",
        "robot_name": ROBOT_NAME,
        "fps": FPS,
        # MANO model metadata
        "mano_flat_hand_mean": False,
        "mano_center_idx": -1,
        "mano_to_robot_scale": 1.0,
        "mano_right_betas": [0.0] * 10,
        "mano_left_betas": [0.0] * 10,
        "mano_link_names": MANO_LINK_NAMES,
        # Object metadata (articulated box: bottom + hinged top)
        "object_name": "box",
        "safe_object_name": "box",
        "object_body_names": ["bottom", "top"],
        "safe_object_body_names": ["bottom", "top"],
        "object_mesh_paths": mesh_paths,
        "object_urdf_paths": [],  # articulated URDF is derived from the mesh path
        "object_mesh_radius": [0.13, 0.10],
    }
    record.update(_object_trajectories())
    record.update(_hand_trajectories("right"))
    record.update(_hand_trajectories("left"))
    return record


def _write_meshes() -> None:
    for name, size in (("bottom", BOTTOM_SIZE), ("top", TOP_SIZE)):
        obj = _box_obj(size)
        for base in (
            OBJECT_ASSETS / "meshes" / "synthbox" / "box",
            ASSET_DIR / "meshes" / "synthbox" / "box",
        ):
            base.mkdir(parents=True, exist_ok=True)
            (base / f"{name}.obj").write_text(obj)


def _write_support_usda() -> None:
    """Write a single support disk under the base box, matching its footprint."""
    hx, hy, _ = (BOTTOM_SIZE / 2.0).tolist()
    cx, cy = float(BOTTOM_CENTER[0]), float(BOTTOM_CENTER[1])
    surface_z = float(BOTTOM_CENTER[2] - BOTTOM_SIZE[2] / 2.0)  # base underside
    radius = max(BOTTOM_SIZE[0], BOTTOM_SIZE[1]) / 2.0
    height = 0.01
    translate_z = surface_z - height / 2.0
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    usda = f"""#usda 1.0
(
    defaultPrim = "support_surfaces"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "support_surfaces"
{{
    def Cylinder "support_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = {height}
        color3f[] primvars:displayColor = [(0.55, 0.78, 0.85)]
        double radius = {radius}
        double3 xformOp:translate = ({cx}, {cy}, {translate_z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
"""
    (RECON_DIR / f"{SEQUENCE_ID}_support.usda").write_text(usda)


def main() -> None:
    """Generate all synthbox fixtures."""
    ManoSharpaData = _load_manosharpadata()

    _write_meshes()
    _write_support_usda()

    empty_ts: dict = {}  # loaded stage leaves robot_* time-series empty
    robot_meta_empty: dict = {
        "right_robot_finger_joint_names": [],
        "right_robot_frame_names": [],
        "right_robot_frame_task_names": [],
        "left_robot_finger_joint_names": [],
        "left_robot_frame_names": [],
        "left_robot_frame_task_names": [],
    }

    # --- loaded fixture (mano + object only) ---
    loaded = _base_record()
    loaded.update(robot_meta_empty)
    loaded.update(empty_ts)
    ManoSharpaData(**loaded).save_to_parquet(
        str(LOADED_DIR), partition_cols=["sequence_id", "robot_name"]
    )

    # --- processed fixture (robot_* filled) ---
    processed = _base_record()
    processed.update(
        {
            "right_robot_finger_joint_names": _finger_joint_names("right"),
            "right_robot_frame_names": _frame_names("right"),
            "right_robot_frame_task_names": _frame_task_names("right"),
            "left_robot_finger_joint_names": _finger_joint_names("left"),
            "left_robot_frame_names": _frame_names("left"),
            "left_robot_frame_task_names": _frame_task_names("left"),
        }
    )
    processed.update(_robot_trajectories("right"))
    processed.update(_robot_trajectories("left"))
    ManoSharpaData(**processed).save_to_parquet(
        str(PROCESSED_DIR), partition_cols=["sequence_id", "robot_name"]
    )

    print(f"Wrote synthbox fixtures under {SYNTHBOX_DIR}")


if __name__ == "__main__":
    main()
