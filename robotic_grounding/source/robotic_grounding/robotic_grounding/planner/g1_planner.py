# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.motion_schema import MotionData, save_motion_parquet
from robotic_grounding.planner.cli import parse_args
from robotic_grounding.planner.motionbricks.inference import MotionInferenceAgent
from robotic_grounding.planner.trajectory import build_interp_trajectory
from robotic_grounding.planner.utils.motion import (
    assemble_hand_contact_fields,
    assemble_object_fields,
    interpolate_robot_motion_data,
)
from robotic_grounding.planner.utils.qpos import (
    LEFT_WRIST,
    RIGHT_WRIST,
    ROOT_HEIGHT,
    build_full_qpos,
    wrist_ee_error_from_qpos,
)
from robotic_grounding.planner.utils.transforms import (
    apply_local_frame_fix,
    transform_reference,
)
from robotic_grounding.planner.utils.validation import (
    assert_motion_parquet_invariants,
    warn_missing_urdf_mesh_deps,
    warn_reference_issues,
)
from robotic_grounding.planner.visualization import visualize
from robotic_grounding.retarget.data_logger import ManoDex3Data
from robotic_grounding.retarget.support_recon import (
    reconstruct_support_for_sequence,
)

# -- Constants ------------------------------------------------------------------

FPS = 30
HOLD_START_S = 5.0
INTERP_DURATION_S = 5.0
HOLD_END_S = 5.0

NOMINAL_JOINTS = {
    22: -0.5,  # left_shoulder_pitch
    23: 0.2,  # left_shoulder_roll
    25: 0.0,  # left_elbow
    29: -0.5,  # right_shoulder_pitch
    30: -0.2,  # right_shoulder_roll
    32: 0.0,  # right_elbow
}

_ASSETS_DIR = Path(__file__).parent / "assets"


# -- Helpers -------------------------------------------------------------------


def _resolve_asset_path_for_output(path: str | None) -> str | None:
    """Re-root stored asset paths under this checkout when possible.

    Returns the original path if it can't be resolved locally; callers should
    warn the user (a downstream consumer will fail on the missing file).
    """
    if not path:
        return path
    marker = "assets/"
    if marker in path:
        suffix = path.rsplit(marker, maxsplit=1)[-1]
        local = Path(ASSET_DIR) / suffix
        if local.exists():
            return str(local)
    if Path(path).exists():
        return path
    print(
        f"  WARNING: asset path {path!r} could not be resolved against the "
        "current workspace; downstream scene-spawn will fail on the missing file."
    )
    return path


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


def _trim_motion_data_range(
    motion: Any, start_frame: int, end_frame: int | None = None
) -> Any:
    """Keep a frame range from every frame-major motion field."""
    if start_frame <= 0 and end_frame is None:
        return motion
    n_frames = len(motion.robot_right_wrist_position)
    start_frame = max(0, int(start_frame))
    end_frame = n_frames if end_frame is None else min(n_frames, int(end_frame))
    if start_frame >= end_frame:
        raise ValueError(
            f"V2P trim range [{start_frame}, {end_frame}) is empty for reference "
            f"length {n_frames}"
        )
    for attr, value in vars(motion).items():
        if isinstance(value, (str, bytes)) or value is None:
            continue
        try:
            value_len = len(value)
        except TypeError:
            continue
        if value_len != n_frames:
            continue
        if isinstance(value, tuple):
            setattr(motion, attr, value[start_frame:end_frame])
        else:
            setattr(motion, attr, value[start_frame:end_frame])
    return motion


def _hand_object_contact_frame_bounds(
    motion: Any, threshold: float = 1e-5
) -> tuple[int | None, int | None]:
    """Return first/last frames with any nonzero hand-object contact point."""
    first: int | None = None
    last: int | None = None
    for side in ("left", "right"):
        contacts = getattr(motion, f"mano_{side}_object_contact_positions", None)
        if not contacts:
            continue
        arr = np.asarray(contacts, dtype=np.float32)
        if arr.ndim < 3 or arr.shape[0] == 0:
            continue
        xyz = arr[..., :3]
        active = np.abs(xyz).sum(axis=-1) > threshold
        frame_ids = np.flatnonzero(active.any(axis=1))
        if frame_ids.size == 0:
            continue
        side_first = int(frame_ids[0])
        side_last = int(frame_ids[-1])
        first = side_first if first is None else min(first, side_first)
        last = side_last if last is None else max(last, side_last)
    return first, last


def _first_hand_object_contact_frame(
    motion: Any, threshold: float = 1e-5
) -> int | None:
    """Return the first frame with any nonzero hand-object contact point."""
    first, _ = _hand_object_contact_frame_bounds(motion, threshold)
    return first


def load_v2p_reference(
    parquet_folder: str,
    filters: list,
    trajectory_id: int = 0,
    target_fps: float = 100.0,
    robot_type: str = "dex3",
    start_frame: int = 0,
    start_at_first_contact: bool = False,
    pre_contact_frames: int = 0,
    end_after_last_contact_frames: int = -1,
) -> dict[str, Any]:
    """Load and interpolate V2P retargeted reference data."""
    if robot_type != "dex3":
        raise ValueError(f"Unsupported robot_type={robot_type!r}; expected 'dex3'.")
    data_class = ManoDex3Data
    motion = data_class.from_parquet(
        root_path=parquet_folder, filters=filters, trajectory_id=trajectory_id
    )
    motion = interpolate_robot_motion_data(motion, target_fps)
    n_frames = len(motion.robot_right_wrist_position)
    effective_start_frame = int(start_frame)
    effective_end_frame = None
    first_contact_frame = None
    last_contact_frame = None
    if start_at_first_contact:
        first_contact_frame, last_contact_frame = _hand_object_contact_frame_bounds(
            motion
        )
        if first_contact_frame is None:
            print("  WARNING: no hand-object contact detected; using v2p_start_frame")
        else:
            contact_start = max(0, first_contact_frame - int(pre_contact_frames))
            effective_start_frame = max(effective_start_frame, contact_start)
    elif int(end_after_last_contact_frames) >= 0:
        first_contact_frame, last_contact_frame = _hand_object_contact_frame_bounds(
            motion
        )
    if int(end_after_last_contact_frames) >= 0:
        if last_contact_frame is None:
            print("  WARNING: no hand-object contact detected; not trimming V2P end")
        else:
            effective_end_frame = min(
                n_frames, last_contact_frame + int(end_after_last_contact_frames) + 1
            )
    motion = _trim_motion_data_range(motion, effective_start_frame, effective_end_frame)
    first_contact_frame_in_reference = None
    if first_contact_frame is not None:
        rel_contact = int(first_contact_frame) - int(effective_start_frame)
        n_trimmed = len(motion.robot_right_wrist_position)
        if 0 <= rel_contact < n_trimmed:
            first_contact_frame_in_reference = rel_contact

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
        "start_frame": effective_start_frame,
        "end_frame": effective_end_frame,
        "first_contact_frame": first_contact_frame,
        "first_contact_frame_in_reference": first_contact_frame_in_reference,
        "last_contact_frame": last_contact_frame,
        "_motion_data": motion,
    }


def save_planner_parquet(
    output_dir: str,
    full_qpos: np.ndarray,
    ref_data: dict[str, Any],
    model: mujoco.MjModel,
    ref_raw: dict[str, Any],
    robot_type: str,
    sequence_id: str,
) -> int:
    """Save planner output as a `motion_v1` Hive-partitioned parquet.

    Embeds: body qpos (from planner), EE/object/finger/contact data (from V2P).
    """
    T = full_qpos.shape[0]
    T_ref = ref_data["left_pos"].shape[0]
    T_use = min(T, T_ref)

    # Decompose full_qpos into root + body + finger slices. Body joints are
    # all non-hand actuated joints regardless of where they sit in the qpos
    # layout — for dex3 the right arm joints come AFTER the left finger
    # joints, so a single contiguous slice would drop them.
    qpos_slice = full_qpos[:T_use]
    robot_root_position = qpos_slice[:, 0:3].tolist()
    robot_root_wxyz = qpos_slice[:, 3:7].tolist()

    # robot_joint_names / robot_joint_positions cover every actuated joint
    # (body + fingers) in MuJoCo joint order. The env's tracking_command
    # resolves cfg.joint_names against this list by name, so fingers must be
    # present or it raises "Motion joint reference is missing tracked robot
    # joints". The per-side `hand_finger_joints` / `hand_finger_joint_names`
    # lists below stay populated for callers that want the side-segregated view.
    body_qpos_idx: list[int] = []
    robot_joint_names: list[str] = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name or "free" in name.lower():
            continue
        body_qpos_idx.append(int(model.jnt_qposadr[i]))
        robot_joint_names.append(name)
    robot_joint_positions = qpos_slice[:, body_qpos_idx].tolist()

    # EE pose built from per-side wrist position + wxyz reference.
    left_pos = np.asarray(ref_data["left_pos"][:T_use], dtype=np.float32)
    left_quat = np.asarray(ref_data["left_quat"][:T_use], dtype=np.float32)
    right_pos = np.asarray(ref_data["right_pos"][:T_use], dtype=np.float32)
    right_quat = np.asarray(ref_data["right_quat"][:T_use], dtype=np.float32)
    ee_pose_w = np.stack(
        [
            np.concatenate([left_pos, left_quat], axis=-1),
            np.concatenate([right_pos, right_quat], axis=-1),
        ],
        axis=1,
    )  # (T, 2, 7)

    # Object body position / wxyz — support single- or multi-body.
    obj_body_pos = ref_data.get("object_pos_all", ref_data["object_pos"][:, None])
    obj_body_wxyz = ref_data.get("object_quat_all", ref_data["object_quat"][:, None])
    obj_body_pos_arr = np.asarray(obj_body_pos, dtype=np.float32)[:T_use]
    obj_body_wxyz_arr = np.asarray(obj_body_wxyz, dtype=np.float32)[:T_use]
    object_body_position = obj_body_pos_arr.tolist()
    object_body_wxyz = obj_body_wxyz_arr.tolist()

    motion = ref_raw.get("_motion_data")
    object_fields = assemble_object_fields(
        motion=motion,
        obj_body_pos_arr=obj_body_pos_arr,
        obj_body_wxyz_arr=obj_body_wxyz_arr,
        T_use=T_use,
        fallback_object_name=str(ref_data.get("object_name", "box")),
        resolve_asset_path=_resolve_asset_path_for_output,
        warn_missing_deps=warn_missing_urdf_mesh_deps,
    )

    hand_fields = assemble_hand_contact_fields(
        motion=motion,
        obj_body_pos_arr=obj_body_pos_arr,
        obj_body_wxyz_arr=obj_body_wxyz_arr,
        T_use=T_use,
    )

    robot_name = "g1_dex3"
    # ee_link_names tells the env which body the EE pose was recorded from.
    # For dex3 the per-side wrist-position fields actually hold the palm-link
    # pose (the free-flyer URDF root), so a `wrist_yaw_link` label would put
    # the env's reward target on the wrong body with a ~4 cm systematic offset.
    ee_link_names = ["left_hand_palm_link", "right_hand_palm_link"]
    md = MotionData(
        sequence_id=sequence_id,
        robot_name=robot_name,
        motion_kind="single_robot",
        source_dataset="planner",
        raw_motion_file="",
        fps=float(FPS),
        coord_frame="robot_base_z_up",
        robot_joint_names=robot_joint_names,
        robot_root_position=robot_root_position,
        robot_root_wxyz=robot_root_wxyz,
        robot_joint_positions=robot_joint_positions,
        ee_link_names=ee_link_names,
        ee_pose_w=ee_pose_w.tolist(),
        object_body_position=object_body_position,
        object_body_wxyz=object_body_wxyz,
        **object_fields,
        **hand_fields,
    )
    # Layout: `<output_dir>/planner_processed/sequence_id=…/robot_name=…/*.parquet`
    # with the support USD as a sibling at `<output_dir>/reconstructed_stage/`.
    # SceneConfig._discover_support_surface walks
    # `<sequence_id_dir>.parent.parent.parent / reconstructed_stage`, so the
    # extra `planner_processed/` layer is what lets it find the support file.
    partition_root = Path(output_dir) / "planner_processed"
    partition_dir = save_motion_parquet(md, root_path=str(partition_root))
    print(f"  Saved {partition_dir} ({T_use} frames)")
    # Hard-fail before training ever sees the parquet so silent data
    # corruption can't make it past planning.
    assert_motion_parquet_invariants(partition_dir, robot_type=robot_type)
    return T_use


def main() -> None:
    """Run the full planner pipeline end-to-end from the CLI args."""
    args = parse_args()
    hold_start_s = 0.0 if args.no_approach else args.hold_start_s
    interp_s = 0.0 if args.no_approach else args.interp_s
    hold_end_s = 0.0 if args.no_approach else args.hold_end_s

    scene_xml = str(_ASSETS_DIR / "mujoco" / "scene_29dof.xml")
    hand_xml = str(_ASSETS_DIR / "mujoco" / "g1_dex3_hands.xml")

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
        start_frame=args.v2p_start_frame,
        start_at_first_contact=args.v2p_start_at_first_contact,
        pre_contact_frames=args.v2p_pre_contact_frames,
        end_after_last_contact_frames=args.v2p_end_after_last_contact_frames,
    )
    if ref_raw.get("first_contact_frame") is not None:
        print(
            "  first hand-object contact: "
            f"frame {ref_raw['first_contact_frame']} "
            f"(keeping {args.v2p_pre_contact_frames} pre-contact frames)"
        )
    if ref_raw.get("last_contact_frame") is not None:
        print(f"  last hand-object contact: frame {ref_raw['last_contact_frame']}")
    if ref_raw.get("start_frame", 0):
        print(f"  dropped leading {ref_raw['start_frame']} interpolated V2P frames")
    if ref_raw.get("end_frame") is not None:
        print(
            "  trimmed V2P end at interpolated frame "
            f"{ref_raw['end_frame']} "
            f"(keeping {args.v2p_end_after_last_contact_frames} post-contact frames)"
        )
    print(f"  {ref_raw['left_pos'].shape[0]} frames at {ref_raw['fps']}fps")
    # Reference-owned checks: warn if input motion or workspace assets are
    # missing fields the planner can't produce on its own. Issues here belong
    # to the upstream retargeting / asset pipeline, not this script.
    warn_reference_issues(ref_raw.get("_motion_data"), ref_raw, args.robot)
    heading_frame = 0
    if args.heading_align_frame == "first_contact":
        first_contact_ref = ref_raw.get("first_contact_frame_in_reference")
        if first_contact_ref is None:
            print(
                "  WARNING: first-contact heading requested but no contact frame is available; using frame 0"
            )
        else:
            heading_frame = int(first_contact_ref)
            print(
                "  heading alignment frame: first contact "
                f"(reference frame {heading_frame})"
            )
    else:
        print("  heading alignment frame: start (reference frame 0)")

    # Step 6: Inference (initial agent build)
    if args.ik_verify or args.ik_plan:
        print("IK mode — not implemented in this script yet")
        return

    print("\nStep 6: Inference")
    agent = MotionInferenceAgent(device="cuda")
    vis_xml = hand_xml if os.path.exists(hand_xml) else scene_xml
    model = mujoco.MjModel.from_xml_path(vis_xml)

    def _plan_one_offset(
        delta_yaw_offset_rad: float,
    ) -> tuple[
        dict[str, Any],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        float,
    ]:
        """Run Steps 3-7 for a single heading offset and return outputs + score.

        Returns ``(ref_data, traj_tuple, qpos_full, mfm_ref, full_qpos,
        ref_start, score)`` where ``score`` is the mean wrist tracking
        error in metres.
        """
        ref_data_local = transform_reference(
            ref_raw,
            nom,
            workspace_offset=tuple(args.workspace_offset),
            robot_type=args.robot,
            delta_yaw_offset=delta_yaw_offset_rad,
            heading_frame=heading_frame,
        )
        T_raw_local = ref_data_local["left_pos"].shape[0]
        N_ref_local = (
            T_raw_local if args.ref_seconds < 0 else int(FPS * args.ref_seconds)
        )
        plan_n_ref_local = min(N_ref_local, T_raw_local)
        traj = build_interp_trajectory(
            nom,
            ref_data_local["left_pos"],
            ref_data_local["left_quat"],
            ref_data_local["right_pos"],
            ref_data_local["right_quat"],
            fps=FPS,
            hold_start_s=hold_start_s,
            interp_s=interp_s,
            hold_end_s=hold_end_s,
            n_ref=plan_n_ref_local,
        )
        traj_lp_l, traj_lq_l, traj_rp_l, traj_rq_l, seg_local = traj
        T_total_local = len(traj_lp_l)
        ref_start_local = seg_local["ref_start"]
        root_pos_local = np.zeros((T_total_local, 3), dtype=np.float32)
        root_pos_local[:, 2] = ROOT_HEIGHT
        root_wxyz_local = np.tile([1.0, 0.0, 0.0, 0.0], (T_total_local, 1)).astype(
            np.float32
        )
        result_local = agent.infer_from_ee_positions(
            root_pos_local,
            root_wxyz_local,
            traj_lp_l,
            traj_lq_l,
            traj_rp_l,
            traj_rq_l,
            root_height_override=ROOT_HEIGHT,
            max_chunk_tokens=6,
            modes=("autoregressive",),
            smooth=not args.no_smooth_qpos,
            half_stride_blend=True,
            fix_lower_body=args.fix_lower_body,
        )
        qpos_full_local = result_local["autoregressive"]["qpos"]
        T_mfm_local = qpos_full_local.shape[0]
        T_ref_mfm_local = T_mfm_local - ref_start_local
        mfm_ref_local = qpos_full_local[
            ref_start_local : ref_start_local + T_ref_mfm_local
        ]
        T_target_local = plan_n_ref_local
        if T_ref_mfm_local != T_target_local and T_ref_mfm_local > 1:
            t_src = np.linspace(0, 1, T_ref_mfm_local)
            t_dst = np.linspace(0, 1, T_target_local)
            pos_joints = np.concatenate(
                [mfm_ref_local[:, :3], mfm_ref_local[:, 7:]], axis=1
            )
            pos_joints_interp = interp1d(t_src, pos_joints, axis=0, kind="linear")(
                t_dst
            )
            root_quats = Rotation.from_quat(mfm_ref_local[:, 3:7])
            root_quats_interp = Slerp(t_src, root_quats)(t_dst).as_quat()
            mfm_ref_new = np.zeros(
                (T_target_local, mfm_ref_local.shape[1]),
                dtype=mfm_ref_local.dtype,
            )
            mfm_ref_new[:, :3] = pos_joints_interp[:, :3]
            mfm_ref_new[:, 3:7] = root_quats_interp
            mfm_ref_new[:, 7:] = pos_joints_interp[:, 3:]
            mfm_ref_local = mfm_ref_new
        else:
            mfm_ref_local = mfm_ref_local[:T_target_local]
        T_save_local = min(T_target_local, mfm_ref_local.shape[0])
        full_qpos_local, _, _, _ = build_full_qpos(
            mfm_ref_local,
            ref_data_local,
            model,
            T_save_local,
            fix_lower_body=args.fix_lower_body,
            fix_root_pos=args.fix_root_pos,
            fix_root_rot=args.fix_root_rot,
            fix_root_z=args.fix_root_z,
            fix_root_rp=args.fix_root_rp,
            fix_root_components=args.fix_root,
        )
        score = wrist_ee_error_from_qpos(full_qpos_local, ref_data_local, model)
        return (
            ref_data_local,
            (traj_lp_l, traj_lq_l, traj_rp_l, traj_rq_l),
            qpos_full_local,
            mfm_ref_local,
            full_qpos_local,
            ref_start_local,
            score,
        )

    if args.search_heading_deg > 0.0:
        n = float(args.search_heading_deg)
        candidates_deg = [-n, -n / 2.0, 0.0, n / 2.0, n]
        print(
            f"\nLocal heading search across {candidates_deg} deg "
            "(picking lowest wrist EE error):"
        )
        scored = []
        for d_deg in candidates_deg:
            res = _plan_one_offset(np.radians(d_deg))
            scored.append((d_deg, res))
            print(f"  Δ={d_deg:+5.1f}°: wrist EE = {res[-1] * 1000:.1f}mm")
        best_deg, best_res = min(scored, key=lambda kv: kv[1][-1])
        print(f"  → best Δ={best_deg:+.1f}° (EE = {best_res[-1] * 1000:.1f}mm)")
        ref_data, traj_tuple, qpos_full, mfm_ref, full_qpos, ref_start, _ = best_res
    else:
        ref_data, traj_tuple, qpos_full, mfm_ref, full_qpos, ref_start, score = (
            _plan_one_offset(0.0)
        )
        print(f"  Wrist EE error: {score * 1000:.1f}mm")
    traj_lp, traj_lq, traj_rp, traj_rq = traj_tuple
    print(f"  Yaw correction: {np.degrees(ref_data['delta_yaw']):.1f} deg")
    print(f"  {full_qpos.shape}")

    # Step 8: Save parquet
    print("\nStep 8: Save parquet")
    sequence_id = args.v2p_sequence
    # Try to extract full sequence ID from the parquet data
    motion = ref_raw.get("_motion_data")
    if motion and hasattr(motion, "sequence_id") and motion.sequence_id:
        sequence_id = motion.sequence_id

    # `output_dir` is the dataset root; the planner writes parquet under
    # `<output_dir>/planner_processed/…` and the support USD under
    # `<output_dir>/reconstructed_stage/…`. SceneConfig discovers the support
    # USD by walking up from the parquet's `sequence_id=…` dir.
    output_dir = args.output or str(Path.cwd() / "planner_output")
    save_planner_parquet(
        output_dir,
        full_qpos,
        ref_data,
        model,
        ref_raw,
        args.robot,
        sequence_id,
    )

    # Step 8b: Reconstruct support surface USD for the transformed object
    # positions. SceneConfig.from_motion_file walks parquet -> ../../../
    # /reconstructed_stage/<seq>_support.usda, so we drop the USD next to
    # the planner output rather than relying on the upstream retarget's
    # un-transformed surface.
    print("\nStep 8b: Reconstruct support surface")
    support_dir = Path(output_dir) / "reconstructed_stage"
    support_dir.mkdir(parents=True, exist_ok=True)
    support_usda = support_dir / f"{sequence_id}_support.usda"
    reconstruct_support_for_sequence(
        input_dir=Path(output_dir) / "planner_processed",
        sequence_id=sequence_id,
        output_override=str(support_usda),
        schema="motion_v1",
    )

    # Step 9: Viewer — show the FULL trajectory (warmup + reference)
    if not args.no_viewer:
        print("\nStep 9: Viewer | Space=pause")
        # Build full qpos for the entire planner output (warmup + reference)
        vis_full_qpos, _, _, _ = build_full_qpos(
            qpos_full,
            ref_data,
            model,
            qpos_full.shape[0],
            fix_lower_body=args.fix_lower_body,
            fix_root_pos=args.fix_root_pos,
            fix_root_rot=args.fix_root_rot,
            fix_root_z=args.fix_root_z,
            fix_root_rp=args.fix_root_rp,
            fix_root_components=args.fix_root,
        )

        # Discover object mesh and support surface from V2P parquet
        object_mesh_path = None
        support_usda_path = None
        resolved_mesh_paths: list[str] = []
        motion = ref_raw.get("_motion_data")
        if motion:
            # Resolve every per-body mesh path declared in the parquet,
            # not just the first that exists locally — multi-body objects
            # (e.g. taco spoon + pan) need all of them rendered.
            mesh_paths = getattr(motion, "object_mesh_paths", None)
            obj_name = getattr(motion, "object_name", None)
            if mesh_paths:
                iterable = mesh_paths if isinstance(mesh_paths, list) else [mesh_paths]
                for mp in iterable:
                    if not mp:
                        continue
                    if os.path.exists(mp):
                        resolved_mesh_paths.append(mp)
                        continue
                    suffix = (
                        mp.split("assets/meshes/")[-1]
                        if "assets/meshes/" in mp
                        else None
                    )
                    if not suffix:
                        continue
                    local = os.path.join(ASSET_DIR, "meshes", suffix)
                    if os.path.exists(local):
                        resolved_mesh_paths.append(local)
                        continue
                    tex = os.path.join(os.path.dirname(local), "mesh_tex.obj")
                    if os.path.exists(tex):
                        resolved_mesh_paths.append(tex)
                if resolved_mesh_paths:
                    object_mesh_path = resolved_mesh_paths[0]

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

        if resolved_mesh_paths:
            print(f"  Object meshes ({len(resolved_mesh_paths)}):")
            for mp in resolved_mesh_paths:
                print(f"    {mp}")
        elif object_mesh_path:
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

        # Per-body mesh list for the viewer. Prefer the resolved per-body
        # mesh_paths from the parquet (one entry per object body); fall back
        # to the arctic registry layout, then to the singular mesh.
        object_mesh_paths_list = list(resolved_mesh_paths)
        if not object_mesh_paths_list:
            body_names = getattr(motion, "object_body_names", None) or []
            obj_name = ref_data.get("object_name", "box")
            mesh_base = os.path.join(ASSET_DIR, "meshes", "arctic", obj_name)
            for bname in body_names:
                bp = os.path.join(mesh_base, f"{bname}.obj")
                if os.path.exists(bp):
                    object_mesh_paths_list.append(bp)
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
