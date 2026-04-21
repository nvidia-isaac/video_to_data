"""G1 whole-body planner: EE targets → full-body motion via learned model.

Generates planned whole-body motion from V2P retargeted hand/object trajectories.
Outputs a Hive-partitioned parquet with body qpos, EE targets, hand keypoints,
contacts, and scene metadata.

Usage:
    MUJOCO_GL=egl python -m robotic_grounding.planner.g1_planner \
        --v2p_parquet /path/to/arctic_processed \
        --v2p_sequence box_grab \
        --output /path/to/planner_processed
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.planner.data_adapters import interpolate_robot_motion_data
from robotic_grounding.planner.inference import MotionInferenceAgent
from robotic_grounding.planner.motion_reps import xyzw_to_wxyz
from robotic_grounding.planner.trajectory import build_interp_trajectory
from robotic_grounding.planner.transforms import (
    apply_local_frame_fix,
    transform_reference,
)
from robotic_grounding.planner.visualization import visualize
from robotic_grounding.retarget.data_logger import ManoDex3Data, ManoSharpaData

# -- Constants ------------------------------------------------------------------

FPS = 30
HOLD_START_S = 5.0
INTERP_DURATION_S = 5.0
HOLD_END_S = 5.0
ROOT_HEIGHT = 0.793

NOMINAL_JOINTS = {
    22: -0.5,  # left_shoulder_pitch
    23: 0.2,  # left_shoulder_roll
    25: 0.0,  # left_elbow
    29: -0.5,  # right_shoulder_pitch
    30: -0.2,  # right_shoulder_roll
    32: 0.0,  # right_elbow
}

LEG_OVERRIDES = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
}

LEFT_WRIST = "left_wrist_yaw_link"
RIGHT_WRIST = "right_wrist_yaw_link"

SUPPORT_SIZE = [0.15, 0.15, 0.952]

# Per-sequence V2P table positions (XY from retargeting, Z=0.475 table center)
V2P_TABLE_POSITIONS = {
    "waffleiron_grab": [-0.0057, -0.0823, 0.475],
    "capsulemachine_grab": [0.0198, -0.0823, 0.475],
    "espressomachine_grab": [0.1075, -0.0941, 0.475],
    "microwave_grab": [-0.0059, -0.1399, 0.475],
    "mixer_grab": [-0.0039, -0.0776, 0.475],
}

_ASSETS_DIR = Path(__file__).parent / "assets"


# -- Helpers -------------------------------------------------------------------


def get_nominal_ee(xml_path: str) -> dict:
    """FK at nominal arm pose to get wrist positions/quats."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[2] = ROOT_HEIGHT
    data.qpos[3] = 1.0  # wxyz identity quaternion w component
    for idx, val in NOMINAL_JOINTS.items():
        data.qpos[idx] = val
    mujoco.mj_forward(model, data)
    left_id = model.body(LEFT_WRIST).id
    right_id = model.body(RIGHT_WRIST).id
    return {
        "left_pos": data.xpos[left_id].copy().astype(np.float32),
        "left_quat": data.xquat[left_id].copy().astype(np.float32),
        "right_pos": data.xpos[right_id].copy().astype(np.float32),
        "right_quat": data.xquat[right_id].copy().astype(np.float32),
    }


def load_v2p_reference(
    parquet_folder, filters, trajectory_id=0, target_fps=100.0, robot_type="sharpa"
):
    """Load and interpolate V2P retargeted reference data."""
    if robot_type == "sharpa":
        data_class = ManoSharpaData
    else:
        data_class = ManoDex3Data
    motion = data_class.from_parquet(
        root_path=parquet_folder, filters=filters, trajectory_id=trajectory_id
    )
    motion = interpolate_robot_motion_data(motion, target_fps)

    obj_pos_all = np.array(motion.object_body_position, dtype=np.float32)  # (T, B, 3)
    obj_quat_all = np.array(motion.object_body_wxyz, dtype=np.float32)  # (T, B, 4)
    # Primary body (body 0) for heading/transform computation
    if obj_pos_all.ndim == 3:
        obj_pos = obj_pos_all[:, 0]  # (T, 3)
        obj_quat = obj_quat_all[:, 0]
    else:
        obj_pos = obj_pos_all
        obj_quat = obj_quat_all

    return {
        "left_pos": np.array(motion.robot_left_wrist_position, dtype=np.float32),
        "left_quat": np.array(motion.robot_left_wrist_wxyz, dtype=np.float32),
        "right_pos": np.array(motion.robot_right_wrist_position, dtype=np.float32),
        "right_quat": np.array(motion.robot_right_wrist_wxyz, dtype=np.float32),
        "left_finger_joints": np.array(
            motion.robot_left_finger_joints, dtype=np.float32
        ),
        "right_finger_joints": np.array(
            motion.robot_right_finger_joints, dtype=np.float32
        ),
        "left_joint_names": list(motion.left_robot_finger_joint_names or []),
        "right_joint_names": list(motion.right_robot_finger_joint_names or []),
        "object_pos": obj_pos,
        "object_quat": obj_quat,
        "object_pos_all": obj_pos_all,
        "object_quat_all": obj_quat_all,
        "object_name": getattr(motion, "object_name", None),
        "object_mesh_paths": getattr(motion, "object_mesh_paths", None),
        "fps": target_fps,
        "_motion_data": motion,
    }


G1_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def build_body_joint_mapping(model):
    """Map 29 body DOFs to combined model qpos indices."""
    mapping = {}
    for dof_idx, jname in enumerate(G1_BODY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            mapping[dof_idx] = int(model.jnt_qposadr[jid])
    return mapping


def build_finger_mapping(model, joint_names):
    """Map finger joint names to model qpos indices."""
    result = []
    for jname in joint_names:
        try:
            result.append(int(model.jnt_qposadr[model.joint(jname).id]))
        except Exception:
            result.append(-1)
    return result


def build_full_qpos(planned_qpos, ref_data, model, T_save):
    """Combine planned body + reference fingers + static legs."""
    nq = model.nq
    full_qpos = np.zeros((T_save, nq), dtype=np.float32)
    body_map = build_body_joint_mapping(model)
    l_finger_map = build_finger_mapping(model, ref_data.get("left_joint_names", []))
    r_finger_map = build_finger_mapping(model, ref_data.get("right_joint_names", []))

    for t in range(T_save):
        t_plan = min(t, planned_qpos.shape[0] - 1)
        full_qpos[t, :3] = planned_qpos[t_plan, :3]
        full_qpos[t, 3:7] = xyzw_to_wxyz(planned_qpos[t_plan, 3:7])
        for dof_idx, qi in body_map.items():
            full_qpos[t, qi] = planned_qpos[t_plan, 7 + dof_idx]
        for jname, val in LEG_OVERRIDES.items():
            try:
                full_qpos[t, int(model.jnt_qposadr[model.joint(jname).id])] = val
            except Exception:
                pass
        f_idx = min(t, ref_data["left_finger_joints"].shape[0] - 1)
        for j, qi in enumerate(l_finger_map):
            if qi >= 0 and j < ref_data["left_finger_joints"].shape[1]:
                full_qpos[t, qi] = ref_data["left_finger_joints"][f_idx, j]
        for j, qi in enumerate(r_finger_map):
            if qi >= 0 and j < ref_data["right_finger_joints"].shape[1]:
                full_qpos[t, qi] = ref_data["right_finger_joints"][f_idx, j]

    return full_qpos, body_map, l_finger_map, r_finger_map


def compute_support_position(ref_raw, ref_data, args):
    """Compute support surface position through the same transform pipeline."""
    seq_key = args.v2p_sequence
    for key in V2P_TABLE_POSITIONS:
        if key in seq_key:
            seq_key = key
            break
    v2p_table_pos = np.array(V2P_TABLE_POSITIONS.get(seq_key, [0.0, -0.14, 0.475]))
    R_yaw = Rotation.from_euler("z", ref_data["delta_yaw"])
    source = apply_local_frame_fix(ref_raw, robot_type=args.robot)
    src_midpoint = 0.5 * (source["left_pos"][0] + source["right_pos"][0])
    table_transformed = R_yaw.apply(v2p_table_pos - src_midpoint) + src_midpoint
    table_transformed += ref_data["offset"]
    table_transformed += np.array(args.workspace_offset)
    return table_transformed.tolist()


def save_planner_parquet(
    output_dir,
    full_qpos,
    ref_data,
    model,
    ref_raw,
    support_pos,
    robot_type,
    sequence_id,
):
    """Save planner output as Hive-partitioned parquet.

    Embeds: body qpos (from planner), EE/object/finger/contact data (from V2P),
    scene metadata (support surface, object info).
    """
    T = full_qpos.shape[0]
    T_ref = ref_data["left_pos"].shape[0]
    T_use = min(T, T_ref)

    # Joint names from model
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    # Build qpos_layout
    nq = int(model.nq)
    left_finger_start = right_finger_start = None
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        adr = int(model.jnt_qposadr[i])
        if "left_hand" in name and left_finger_start is None:
            left_finger_start = adr
        if "right_hand" in name and right_finger_start is None:
            right_finger_start = adr
    body_end = left_finger_start if left_finger_start else nq
    left_finger_end = right_finger_start if right_finger_start else nq
    right_finger_end = nq
    qpos_layout = json.dumps(
        {
            "root_pos": [0, 3],
            "root_quat_wxyz": [3, 7],
            "body_joints": [7, body_end],
            "left_finger_joints": (
                [left_finger_start, left_finger_end] if left_finger_start else [nq, nq]
            ),
            "right_finger_joints": (
                [right_finger_start, right_finger_end]
                if right_finger_start
                else [nq, nq]
            ),
        }
    )

    # Build data dict for parquet
    data = {
        # Planner body qpos
        "qpos": [full_qpos[:T_use].tolist()],
        "joint_names": [joint_names],
        "qpos_layout": [qpos_layout],
        "fps": [float(FPS)],
        "ee_link_names": [["left_wrist_yaw_link", "right_wrist_yaw_link"]],
        # Wrist targets from reference (per-side)
        "robot_left_wrist_position": [ref_data["left_pos"][:T_use].tolist()],
        "robot_left_wrist_wxyz": [ref_data["left_quat"][:T_use].tolist()],
        "robot_right_wrist_position": [ref_data["right_pos"][:T_use].tolist()],
        "robot_right_wrist_wxyz": [ref_data["right_quat"][:T_use].tolist()],
        # Object from reference (all bodies)
        "object_body_position": [
            ref_data.get("object_pos_all", ref_data["object_pos"][:, None])[
                :T_use
            ].tolist()
        ],
        "object_body_wxyz": [
            ref_data.get("object_quat_all", ref_data["object_quat"][:, None])[
                :T_use
            ].tolist()
        ],
        "object_name": [ref_data.get("object_name", "box")],
        # Support surface
        "support_position": [support_pos],
        "support_size": [SUPPORT_SIZE],
    }

    # Embed V2P hand/contact data from the original motion_data
    motion = ref_raw.get("_motion_data")
    if motion is not None:
        for side in ("left", "right"):
            wrist_pos = getattr(motion, f"robot_{side}_wrist_position", None)
            wrist_wxyz = getattr(motion, f"robot_{side}_wrist_wxyz", None)
            finger_joints = getattr(motion, f"robot_{side}_finger_joints", None)
            frames = getattr(motion, f"robot_{side}_frames", None)
            frame_names = getattr(motion, f"{side}_robot_frame_names", None)
            link_contacts = getattr(motion, f"mano_{side}_link_contact_positions", None)
            obj_contacts = getattr(
                motion, f"mano_{side}_object_contact_positions", None
            )

            if wrist_pos is not None:
                data[f"robot_{side}_wrist_position"] = [
                    np.array(wrist_pos, dtype=np.float32)[:T_use].tolist()
                ]
            if wrist_wxyz is not None:
                data[f"robot_{side}_wrist_wxyz"] = [
                    np.array(wrist_wxyz, dtype=np.float32)[:T_use].tolist()
                ]
            if finger_joints is not None:
                data[f"robot_{side}_finger_joints"] = [
                    np.array(finger_joints, dtype=np.float32)[:T_use].tolist()
                ]
            if frames is not None:
                data[f"robot_{side}_frames"] = [
                    np.array(frames, dtype=np.float32)[:T_use].tolist()
                ]
            if frame_names is not None:
                data[f"{side}_robot_frame_names"] = [list(frame_names)]
            if link_contacts is not None:
                data[f"mano_{side}_link_contact_positions"] = [
                    np.array(link_contacts, dtype=np.float32)[:T_use].tolist()
                ]
            if obj_contacts is not None:
                data[f"mano_{side}_object_contact_positions"] = [
                    np.array(obj_contacts, dtype=np.float32)[:T_use].tolist()
                ]

    # SceneConfig compatibility columns (needed by SceneConfig.from_motion_file())
    if motion is not None:
        # Object metadata
        for attr in (
            "object_body_names",
            "safe_object_body_names",
            "safe_object_name",
            "object_urdf_paths",
            "object_mesh_paths",
        ):
            val = getattr(motion, attr, None)
            if val is not None:
                data[attr] = [list(val) if isinstance(val, (list, tuple)) else val]

        # Object articulation (needed for articulated/rigid detection)
        obj_art = getattr(motion, "object_articulation", None)
        if obj_art is not None:
            data["object_articulation"] = [
                np.array(obj_art, dtype=np.float32)[:T_use].tolist()
            ]
        else:
            data["object_articulation"] = [[0.0] * T_use]

        # Object root pose (for articulated objects)
        for attr in ("object_root_position", "object_root_axis_angle"):
            val = getattr(motion, attr, None)
            if val is not None:
                data[attr] = [np.array(val, dtype=np.float32)[:T_use].tolist()]

        # Contact normals and part IDs
        for side in ("left", "right"):
            for suffix in ("object_contact_normals", "object_contact_part_ids"):
                attr = f"mano_{side}_{suffix}"
                val = getattr(motion, attr, None)
                if val is not None:
                    data[attr] = [np.array(val)[:T_use].tolist()]

        # Finger joint names
        for side_attr in (
            "left_robot_finger_joint_names",
            "right_robot_finger_joint_names",
        ):
            val = getattr(motion, side_attr, None)
            if val is not None:
                data[side_attr] = [list(val)]

    # Write as Hive-partitioned parquet
    robot_name = "g1_sharpa" if robot_type == "sharpa" else "g1_dex3"
    partition_dir = (
        Path(output_dir) / f"sequence_id={sequence_id}" / f"robot_name={robot_name}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    out_path = partition_dir / "data.parquet"

    table = pa.table(data)
    pq.write_table(table, str(out_path))
    print(f"  Saved {out_path} ({T_use} frames)")
    return T_use


# -- CLI -----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="G1 whole-body planner")
    parser.add_argument("--robot", choices=["sharpa", "dex3"], default="sharpa")
    parser.add_argument("--v2p_parquet", required=True)
    parser.add_argument("--v2p_robot_name", default="sharpa_wave")
    parser.add_argument("--v2p_sequence", default="box_grab")
    parser.add_argument("--v2p_trajectory_id", type=int, default=0)
    parser.add_argument("--target_fps", type=float, default=100.0)
    parser.add_argument(
        "--workspace_offset", type=float, nargs=3, default=[-0.10, 0.0, -0.15]
    )
    parser.add_argument("--ref_seconds", type=float, default=-1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--ik_verify", action="store_true")
    parser.add_argument("--ik_plan", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    scene_xml = str(_ASSETS_DIR / "mujoco" / "scene_29dof.xml")
    hand_xml = str(
        _ASSETS_DIR
        / "mujoco"
        / ("g1_dex3_hands.xml" if args.robot == "dex3" else "g1_sharpa_hands.xml")
    )

    # Step 1: Nominal EE FK
    print(f"Step 1: Nominal FK (robot={args.robot})")
    nom = get_nominal_ee(scene_xml)
    print(
        f"  Left: {np.round(nom['left_pos'], 3)}  Right: {np.round(nom['right_pos'], 3)}"
    )

    # Step 2: Load V2P reference
    print("\nStep 2: Load V2P reference")
    filters = [
        ("robot_name", "=", args.v2p_robot_name),
        ("sequence_id", "contains", args.v2p_sequence),
    ]
    ref_raw = load_v2p_reference(
        args.v2p_parquet,
        filters,
        trajectory_id=args.v2p_trajectory_id,
        target_fps=args.target_fps,
        robot_type=args.robot,
    )
    print(f"  {ref_raw['left_pos'].shape[0]} frames at {ref_raw['fps']}fps")

    # Step 3-4: Transform to G1 frame
    print("\nStep 3-4: Transform to G1 frame")
    ref_data = transform_reference(
        ref_raw,
        nom,
        workspace_offset=tuple(args.workspace_offset),
        robot_type=args.robot,
    )
    print(f"  Yaw correction: {np.degrees(ref_data['delta_yaw']):.1f} deg")

    T_raw = ref_data["left_pos"].shape[0]
    N_ref = T_raw if args.ref_seconds < 0 else int(FPS * args.ref_seconds)
    plan_n_ref = min(N_ref, T_raw)

    # Step 5: Build trajectory
    print(f"\nStep 5: Build trajectory ({plan_n_ref} ref frames)")
    traj_lp, traj_lq, traj_rp, traj_rq, seg = build_interp_trajectory(
        nom,
        ref_data["left_pos"],
        ref_data["left_quat"],
        ref_data["right_pos"],
        ref_data["right_quat"],
        fps=FPS,
        hold_start_s=HOLD_START_S,
        interp_s=INTERP_DURATION_S,
        hold_end_s=HOLD_END_S,
        n_ref=plan_n_ref,
    )
    T_total = len(traj_lp)
    ref_start = seg["ref_start"]
    print(f"  {T_total} frames (ref starts at {ref_start})")

    # Step 6: Inference
    if args.ik_verify or args.ik_plan:
        print("IK mode — not implemented in this script yet")
        return

    print("\nStep 6: Inference")
    agent = MotionInferenceAgent(device="cuda")
    root_pos = np.zeros((T_total, 3), dtype=np.float32)
    root_pos[:, 2] = ROOT_HEIGHT
    root_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (T_total, 1)).astype(np.float32)

    result = agent.infer_from_ee_positions(
        root_pos,
        root_wxyz,
        traj_lp,
        traj_lq,
        traj_rp,
        traj_rq,
        root_height_override=ROOT_HEIGHT,
        max_chunk_tokens=6,
        modes=("autoregressive",),
        smooth=True,
        half_stride_blend=True,
    )
    qpos_full = result["autoregressive"]["qpos"]
    print(f"  Output: {qpos_full.shape}")

    # Extract reference portion and interpolate to match ref frame count
    T_mfm = qpos_full.shape[0]
    T_ref_mfm = T_mfm - ref_start
    mfm_ref = qpos_full[ref_start : ref_start + T_ref_mfm]

    T_target = plan_n_ref
    if T_ref_mfm != T_target and T_ref_mfm > 1:
        print(f"  Interpolating: {T_ref_mfm} → {T_target} frames")
        t_src = np.linspace(0, 1, T_ref_mfm)
        t_dst = np.linspace(0, 1, T_target)
        # Positions + joints: linear
        pos_joints = np.concatenate([mfm_ref[:, :3], mfm_ref[:, 7:]], axis=1)
        pos_joints_interp = interp1d(t_src, pos_joints, axis=0, kind="linear")(t_dst)
        # Root quaternion (xyzw): Slerp
        root_quats = Rotation.from_quat(mfm_ref[:, 3:7])
        root_quats_interp = Slerp(t_src, root_quats)(t_dst).as_quat()
        # Reassemble
        mfm_ref_new = np.zeros((T_target, mfm_ref.shape[1]), dtype=mfm_ref.dtype)
        mfm_ref_new[:, :3] = pos_joints_interp[:, :3]
        mfm_ref_new[:, 3:7] = root_quats_interp
        mfm_ref_new[:, 7:] = pos_joints_interp[:, 3:]
        mfm_ref = mfm_ref_new
    else:
        mfm_ref = mfm_ref[:T_target]
    T_save = min(T_target, mfm_ref.shape[0])

    # Step 7: Build full qpos
    print("\nStep 7: Build full qpos")
    vis_xml = hand_xml if os.path.exists(hand_xml) else scene_xml
    model = mujoco.MjModel.from_xml_path(vis_xml)
    full_qpos, _, _, _ = build_full_qpos(mfm_ref, ref_data, model, T_save)
    print(f"  {full_qpos.shape}")

    # Compute support position
    support_pos = compute_support_position(ref_raw, ref_data, args)
    print(f"  Support: {np.round(support_pos, 3)}")

    # Step 8: Save parquet
    print("\nStep 8: Save parquet")
    sequence_id = args.v2p_sequence
    # Try to extract full sequence ID from the parquet data
    motion = ref_raw.get("_motion_data")
    if motion and hasattr(motion, "sequence_id") and motion.sequence_id:
        sequence_id = motion.sequence_id

    output_dir = args.output or str(Path.cwd() / "planner_processed")
    save_planner_parquet(
        output_dir,
        full_qpos,
        ref_data,
        model,
        ref_raw,
        support_pos,
        args.robot,
        sequence_id,
    )

    # Step 9: Viewer — show the FULL trajectory (warmup + reference)
    if not args.no_viewer:
        print("\nStep 9: Viewer | Space=pause")
        # Build full qpos for the entire planner output (warmup + reference)
        vis_full_qpos, _, _, _ = build_full_qpos(
            qpos_full, ref_data, model, qpos_full.shape[0]
        )

        # Discover object mesh and support surface from V2P parquet
        object_mesh_path = None
        support_usda_path = None
        motion = ref_raw.get("_motion_data")
        if motion:
            # Discover object mesh — try parquet mesh_paths then registry
            mesh_paths = getattr(motion, "object_mesh_paths", None)
            obj_name = getattr(motion, "object_name", None)
            if mesh_paths:
                for mp in (
                    mesh_paths if isinstance(mesh_paths, list) else [mesh_paths]
                ):
                    if os.path.exists(mp):
                        object_mesh_path = mp
                        break
                    # Resolve Docker /workspace/ paths to local assets
                    suffix = (
                        mp.split("assets/meshes/")[-1]
                        if "assets/meshes/" in mp
                        else None
                    )
                    if suffix:
                        local = os.path.join(ASSET_DIR, "meshes", suffix)
                        if os.path.exists(local):
                            object_mesh_path = local
                            break
                        # Try mesh_tex.obj in same directory
                        local_dir = os.path.dirname(local)
                        tex = os.path.join(local_dir, "mesh_tex.obj")
                        if os.path.exists(tex):
                            object_mesh_path = tex
                            break

            # Fallback: look up from object registry
            if object_mesh_path is None and obj_name:
                mesh_dir = os.path.join(ASSET_DIR, "meshes", "arctic", obj_name)
                tex = os.path.join(mesh_dir, "mesh_tex.obj")
                if os.path.exists(tex):
                    object_mesh_path = tex

            # Discover support USDA — walk up from parquet to find reconstructed_stage/
            v2p_path = Path(args.v2p_parquet).resolve()
            if sequence_id:
                for parent in [v2p_path] + list(v2p_path.parents):
                    candidate = (
                        parent / "reconstructed_stage" / f"{sequence_id}_support.usda"
                    )
                    if candidate.exists():
                        support_usda_path = str(candidate)
                        break

        if object_mesh_path:
            print(f"  Object mesh: {object_mesh_path}")
        if support_usda_path:
            print(f"  Support: {support_usda_path}")

        # Build support transform: same yaw + offset as applied to EE/object
        support_xform = None
        if support_usda_path:
            source = apply_local_frame_fix(ref_raw, robot_type=args.robot)
            src_midpoint = 0.5 * (source["left_pos"][0] + source["right_pos"][0])
            support_xform = {
                "delta_yaw": ref_data["delta_yaw"],
                "offset": ref_data["offset"],
                "src_midpoint": src_midpoint,
            }

        obj_pos_vis = ref_data.get("object_pos_all", ref_data.get("object_pos"))
        obj_quat_vis = ref_data.get("object_quat_all", ref_data.get("object_quat"))

        # Resolve per-body meshes from motion data
        object_mesh_paths_list = []
        body_names = getattr(motion, "object_body_names", None) or []
        obj_name = ref_data.get("object_name", "box")
        mesh_base = os.path.join(ASSET_DIR, "meshes", "arctic", obj_name)
        for bname in body_names:
            bp = os.path.join(mesh_base, f"{bname}.obj")
            if os.path.exists(bp):
                object_mesh_paths_list.append(bp)
        # Fallback to single mesh_tex.obj
        if not object_mesh_paths_list and object_mesh_path:
            object_mesh_paths_list = [object_mesh_path]

        visualize(
            vis_xml,
            vis_full_qpos,
            traj_lp,
            traj_lq,
            traj_rp,
            traj_rq,
            left_finger_joints=ref_data["left_finger_joints"],
            right_finger_joints=ref_data["right_finger_joints"],
            recorded_left_names=ref_data["left_joint_names"],
            recorded_right_names=ref_data["right_joint_names"],
            object_pos=obj_pos_vis,
            object_quat=obj_quat_vis,
            fps=FPS,
            ref_start=ref_start,
            object_mesh_paths=object_mesh_paths_list,
            support_usda_path=support_usda_path,
            support_transform=support_xform,
        )


if __name__ == "__main__":
    main()
