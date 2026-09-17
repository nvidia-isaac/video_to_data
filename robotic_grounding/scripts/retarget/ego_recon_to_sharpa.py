# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retarget ego reconstruction loaded data (ManoSharpaData) to Sharpa Wave.

Reads the loaded Parquet (ManoSharpaData with MANO + object only;
produced upstream by reconstruction's v2d_task_library_loader ego_recon loader),
runs IK per frame to fill robot_* fields, and saves to processed.

Object meshes are loaded generically from the Parquet's ``object_mesh_paths``
(a single metric mesh per body), so no dataset-specific mesh handling is needed.

Usage:
  1. (upstream) ego_recon loader writes loaded Parquet
  2. python scripts/retarget/ego_recon_to_sharpa.py --save
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import viser
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import (
    ManoSharpaData,
    add_sequence_filter_args,
    filter_sequence_ids,
    list_sequence_ids,
)
from robotic_grounding.retarget.dataset_registry import get_dataset_config
from robotic_grounding.retarget.object_collision import make_object_projector
from robotic_grounding.retarget.retarget_utils import (
    DEFAULT_PARTITION_COLS,
    run_frame_ik,
    setup_sharpa_kinematics,
    wrist_pose_from_mano_joint0,
)
from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)

_EGO_RECON_CONFIG = get_dataset_config("ego_recon")
# The loaded Parquet is a regenerable intermediate and lives outside the committed
# asset tree; only the processed output belongs under HUMAN_MOTION_DATA_DIR.
DEFAULT_INPUT_DIR = _EGO_RECON_CONFIG.loaded_data_dir
DEFAULT_OUTPUT_DIR = (
    HUMAN_MOTION_DATA_DIR / _EGO_RECON_CONFIG.name / _EGO_RECON_CONFIG.processed_dirname
)

# Ego/WiLoR hands have no wrist link-to-site rotation offset (same as Hot3D/OakInk2).
EGO_WILOR_LINK_TO_SITE_QUAT_XYZW = None


def _load_object_viser_handles(
    viser_server: viser.ViserServer,
    object_mesh_paths: list[str],
    object_body_names: list[str],
) -> dict[str, Any]:
    """Load object meshes from stored schema paths and add to viser scene.

    Returns dict mapping body name to viser mesh handle.
    """
    handles: dict[str, Any] = {}
    for part, path in zip(object_body_names, object_mesh_paths, strict=True):
        if not path or not Path(path).exists():
            continue
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        handles[part] = viser_server.scene.add_mesh_trimesh(
            name=f"/object/{part}",
            mesh=mesh,
            position=np.array([0.0, 0.0, 0.0]),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    return handles


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ego_recon-to-Sharpa retargeting script."""
    parser = argparse.ArgumentParser(
        description="Retarget ego_recon loaded Parquet data to Sharpa (run IK, fill robot_*)."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--mano_to_robot_scale", type=float, default=1.2)
    parser.add_argument(
        "--hand_object_offset",
        type=float,
        default=0.0,
        help="Constant lateral offset (meters) pushing each hand outward, away from "
        "the object center in the horizontal plane. Compensates for source MANO "
        "grasps that penetrate the object collider (which otherwise makes the sim "
        "contact solver eject the hands).",
    )
    parser.add_argument(
        "--adaptive_offset",
        action="store_true",
        default=False,
        help="Penetration-aware outward offset: per-frame magnitude alpha * smoothed "
        "penetration depth (capped), along the same horizontal-outward direction as "
        "--hand_object_offset. Keeps the constant offset's smooth direction but adapts "
        "to the actual per-frame penetration (zero outside contact). Mutually "
        "exclusive with a nonzero --hand_object_offset.",
    )
    parser.add_argument(
        "--adaptive_offset_alpha",
        type=float,
        default=1.2,
        help="Multiplier on the smoothed penetration depth for --adaptive_offset.",
    )
    parser.add_argument(
        "--adaptive_offset_cap",
        type=float,
        default=0.03,
        help="Maximum outward offset magnitude (meters) for --adaptive_offset.",
    )
    parser.add_argument(
        "--adaptive_offset_smooth_window",
        type=int,
        default=9,
        help="Savitzky-Golay window (frames, odd) smoothing the per-frame penetration "
        "depth before scaling — removes recon jitter from the offset magnitude.",
    )
    parser.add_argument(
        "--smooth_reference",
        action="store_true",
        default=False,
        help="Savitzky-Golay-smooth the MANO joint positions and orientations over "
        "time BEFORE offsets/IK. The per-frame WiLoR reconstruction carries "
        "frame-to-frame jitter (finger joint velocity p95 ~7-9 rad/s with "
        "single-frame spikes to ~50 rad/s on box_on_table_geocalib) that the IK "
        "faithfully transfers into the retargeted reference — which trains as the "
        "action feedforward. The smoothed MANO is also persisted to the output.",
    )
    parser.add_argument(
        "--smooth_reference_window",
        type=int,
        default=9,
        help="Savgol window (frames, odd) for --smooth_reference (9 @ 30 fps = 0.3 s; "
        "well under the ~1 s grasp-closure timescale).",
    )
    parser.add_argument(
        "--smooth_reference_polyorder",
        type=int,
        default=2,
        help="Savgol polynomial order for --smooth_reference.",
    )
    parser.add_argument(
        "--surface_project",
        action="store_true",
        default=False,
        help="Push MANO joint IK targets that penetrate the object out onto its "
        "surface before IK (de-penetration). Adapts per-frame, unlike "
        "--hand_object_offset.",
    )
    parser.add_argument(
        "--surface_margin",
        type=float,
        default=0.005,
        help="Clearance (meters) beyond the object surface when --surface_project "
        "(roughly the robot link radius).",
    )
    parser.add_argument(
        "--surface_granularity",
        type=str,
        default="finger",
        choices=["finger", "hand", "joint"],
        help="Rigidity of the push-out: 'finger' (per-finger rigid, preserves curl; "
        "default), 'hand' (whole hand rigid, exact shape), 'joint' (per-joint, max "
        "contact but distorts shape).",
    )
    parser.add_argument(
        "--surface_method",
        type=str,
        default="obb",
        help="Object collision proxy for projection (only 'obb' implemented).",
    )
    add_sequence_filter_args(parser)
    return parser.parse_args()


def _smooth_mano_reference(data: Any, window: int, polyorder: int) -> None:
    """Savgol-smooth MANO joint positions/orientations over time, in place.

    Positions ``mano_{side}_joints`` (T, 21, 3) are filtered per coordinate along
    time. Orientations ``mano_{side}_joints_wxyz`` (T, 21, 4) are first
    hemisphere-aligned frame-to-frame (q and -q are the same rotation; savgol
    across a sign flip would swing through zero), then filtered and renormalized
    — valid for the small frame-to-frame jitter this targets. Mutating ``data``
    means the IK targets, adaptive-offset depths, and the persisted mano columns
    all see the same smoothed reference.
    """
    from scipy.signal import savgol_filter  # noqa: PLC0415

    for side in ("right", "left"):
        joints = np.asarray(getattr(data, f"mano_{side}_joints"), dtype=np.float64)
        num_frames = joints.shape[0]
        w = min(window if window % 2 == 1 else window + 1, num_frames)
        if w < 5:
            return
        smoothed = savgol_filter(joints, window_length=w, polyorder=polyorder, axis=0)
        # Lists, not ndarrays: keeps to_dict()/save_to_parquet serialization
        # identical to the loaded (as_py) representation.
        setattr(data, f"mano_{side}_joints", smoothed.tolist())
        wxyz = np.asarray(
            getattr(data, f"mano_{side}_joints_wxyz"), dtype=np.float64
        ).copy()
        for t in range(1, num_frames):
            flip = (wxyz[t] * wxyz[t - 1]).sum(axis=-1) < 0.0
            wxyz[t, flip] *= -1.0
        wxyz = savgol_filter(wxyz, window_length=w, polyorder=polyorder, axis=0)
        wxyz /= np.linalg.norm(wxyz, axis=-1, keepdims=True).clip(min=1e-9)
        setattr(data, f"mano_{side}_joints_wxyz", wxyz.tolist())


def _outward_offset(
    joints_all: list, object_root_position: list, dist: float
) -> np.ndarray:
    """Return a constant horizontal outward offset of magnitude ``dist``.

    Points from the object center to a hand's mean wrist (joint 0), i.e.
    outward/sideways from the object.
    """
    if dist == 0.0:
        return np.zeros(3, dtype=np.float32)
    obj_xy = np.asarray(object_root_position, dtype=np.float64).mean(axis=0)[:2]
    wrist_xy = np.asarray([frame[0] for frame in joints_all], dtype=np.float64).mean(
        axis=0
    )[:2]
    direction = wrist_xy - obj_xy
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    direction = direction / norm
    return np.array([direction[0] * dist, direction[1] * dist, 0.0], dtype=np.float32)


def _adaptive_outward_offsets(
    projector: Any,
    joints_all: list,
    object_body_position: list,
    object_body_wxyz: list,
    outward_dir: np.ndarray,
    alpha: float,
    cap: float,
    window: int,
) -> np.ndarray:
    """Per-frame outward offsets scaled by the smoothed penetration depth.

    Keeps the constant offset's proven smooth horizontal-outward *direction* but
    adapts the *magnitude* per frame: ``offset_t = dir * clip(alpha * d_t, 0, cap)``
    where ``d_t`` is the hand's max penetration depth into the object OBB at frame
    ``t``, Savitzky-Golay-smoothed over ``window`` frames so recon jitter does not
    leak into the IK targets. Zero outside contact — unlike the constant offset,
    approach/retreat phases are left untouched.

    Args:
        projector: ``SurfaceProjector`` built from the object meshes.
        joints_all: Per-frame MANO joints, each ``(21, 3)``.
        object_body_position: Per-frame object body positions ``(num_bodies, 3)``.
        object_body_wxyz: Per-frame object body quaternions ``(num_bodies, 4)``.
        outward_dir: Unit horizontal direction (from ``_outward_offset`` at dist=1).
        alpha: Multiplier on the smoothed depth.
        cap: Maximum offset magnitude (meters).
        window: Savgol window length (frames; forced odd, clipped to the sequence).

    Returns:
        Array of shape ``(num_frames, 3)`` with the per-frame offset vectors.
    """
    from scipy.signal import savgol_filter  # noqa: PLC0415

    num_frames = len(joints_all)
    depths = np.zeros(num_frames, dtype=np.float64)
    for t in range(num_frames):
        depths[t] = float(
            projector.penetration(
                np.asarray(joints_all[t], dtype=np.float64),
                np.asarray(object_body_position[t], dtype=np.float64),
                np.asarray(object_body_wxyz[t], dtype=np.float64),
            ).max()
        )
    w = min(window if window % 2 == 1 else window + 1, num_frames)
    if w >= 5:
        depths = savgol_filter(depths, window_length=w, polyorder=2)
    magnitudes = np.clip(alpha * depths, 0.0, cap)
    return (magnitudes[:, None] * outward_dir[None, :]).astype(np.float32)


def main(args: argparse.Namespace) -> None:
    """Read loaded ego_recon Parquet, run IK per frame, save retargeted Parquet."""
    device = torch.device(args.device)

    if args.visualize:
        viser_server = viser.ViserServer()

    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    right_sharpa_kinematics = setup_sharpa_kinematics(
        side="right", frame_tasks_converged_threshold=1e-6
    )
    left_sharpa_kinematics = setup_sharpa_kinematics(
        side="left", frame_tasks_converged_threshold=1e-6
    )

    sequence_ids = list_sequence_ids(str(args.input_dir))
    sequence_ids = filter_sequence_ids(sequence_ids, args)
    print(f"Found {len(sequence_ids)} sequences in {args.input_dir}")

    link_to_site_xyzw = EGO_WILOR_LINK_TO_SITE_QUAT_XYZW

    viser_object_handles: dict[str, Any] = {}
    for sequence_id in tqdm(sequence_ids):
        if args.visualize:
            for handle in viser_object_handles.values():
                handle.remove()
            viser_object_handles.clear()

        data = ManoSharpaData.from_parquet(
            str(args.input_dir),
            filters=[("sequence_id", "=", sequence_id)],
        )
        num_frames = len(data.mano_right_trans)

        # Temporal smoothing FIRST: offsets, projection depths and IK must all see
        # the same (smoothed) reference, and the smoothed mano persists to output.
        if args.smooth_reference:
            _smooth_mano_reference(
                data, args.smooth_reference_window, args.smooth_reference_polyorder
            )

        # Optional per-frame surface projection (de-penetration). Build one OBB proxy
        # per object body from the stored meshes; scaled by the dataset's vertex scale.
        # The projector is also needed by --adaptive_offset for depth measurement.
        projector = None
        if args.surface_project or args.adaptive_offset:
            vertex_scale = get_dataset_config("ego_recon").mesh_vertex_scale
            projector = make_object_projector(
                list(data.object_mesh_paths),
                vertex_scale=vertex_scale,
                method=args.surface_method,
            )

        # Constant per-hand outward (sideways) offset to pull the hands off the
        # object collider — the source MANO grasps penetrate the box, so without
        # this the contact solver ejects the hands in sim.
        right_offset_t = torch.tensor(
            _outward_offset(
                data.mano_right_joints,
                data.object_root_position,
                args.hand_object_offset,
            ),
            dtype=torch.float32,
            device=device,
        )
        left_offset_t = torch.tensor(
            _outward_offset(
                data.mano_left_joints,
                data.object_root_position,
                args.hand_object_offset,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Penetration-aware per-frame offsets (see _adaptive_outward_offsets).
        right_offsets_seq = None
        left_offsets_seq = None
        if args.adaptive_offset:
            if args.hand_object_offset:
                raise SystemExit(
                    "--adaptive_offset is mutually exclusive with a nonzero "
                    "--hand_object_offset (it replaces the constant magnitude)."
                )
            offsets = {}
            for side, joints_all in (
                ("right", data.mano_right_joints),
                ("left", data.mano_left_joints),
            ):
                direction = _outward_offset(
                    joints_all, data.object_root_position, 1.0
                ).astype(np.float64)
                offsets[side] = _adaptive_outward_offsets(
                    projector,
                    joints_all,
                    data.object_body_position,
                    data.object_body_wxyz,
                    direction,
                    args.adaptive_offset_alpha,
                    args.adaptive_offset_cap,
                    args.adaptive_offset_smooth_window,
                )
                mags = np.linalg.norm(offsets[side], axis=-1)
                print(
                    f"[adaptive_offset] {sequence_id} {side}: "
                    f"mean={mags.mean():.4f} max={mags.max():.4f} m "
                    f"({(mags > 1e-6).mean():.0%} frames nonzero)"
                )
            right_offsets_seq = torch.tensor(
                offsets["right"], dtype=torch.float32, device=device
            )
            left_offsets_seq = torch.tensor(
                offsets["left"], dtype=torch.float32, device=device
            )

        if args.visualize:
            object_mesh_paths = getattr(data, "object_mesh_paths", None) or []
            viser_object_handles = _load_object_viser_handles(
                viser_server,
                object_mesh_paths,
                data.object_body_names,
            )

        # Collect IK results per frame
        robot_right_wrist_position = []
        robot_right_wrist_wxyz = []
        robot_right_finger_joints = []
        robot_right_frames = []
        robot_right_frame_task_errors = []
        robot_right_num_optimization_iterations = []
        robot_left_wrist_position = []
        robot_left_wrist_wxyz = []
        robot_left_finger_joints = []
        robot_left_frames = []
        robot_left_frame_task_errors = []
        robot_left_num_optimization_iterations = []

        right_qpos = None
        left_qpos = None

        # Per-frame robot configs captured for the interactive viser scrubber
        # (only populated when --visualize; see the frame-slider block below).
        right_qpos_frames: list[np.ndarray] = []
        left_qpos_frames: list[np.ndarray] = []

        for t in range(num_frames):
            # Per-frame offset: adaptive sequence when enabled, else the constant.
            right_off = (
                right_offsets_seq[t]
                if right_offsets_seq is not None
                else right_offset_t
            )
            left_off = (
                left_offsets_seq[t] if left_offsets_seq is not None else left_offset_t
            )
            if args.surface_project:
                if projector is None:
                    raise RuntimeError(
                        "--surface_project needs an object projector, which requires "
                        "object meshes in the loaded parquet."
                    )
                # Per-frame surface projection (multi-body safe). Optionally apply the
                # outward offset first, then guarantee non-penetration.
                obj_pos = np.asarray(data.object_body_position[t], dtype=np.float64)
                obj_wxyz = np.asarray(data.object_body_wxyz[t], dtype=np.float64)
                rj_np = np.asarray(data.mano_right_joints[t], dtype=np.float64)
                lj_np = np.asarray(data.mano_left_joints[t], dtype=np.float64)
                rj_np = rj_np + right_off.cpu().numpy()
                lj_np = lj_np + left_off.cpu().numpy()
                rj_np = projector.project_out(
                    rj_np,
                    obj_pos,
                    obj_wxyz,
                    args.surface_margin,
                    args.surface_granularity,
                )
                lj_np = projector.project_out(
                    lj_np,
                    obj_pos,
                    obj_wxyz,
                    args.surface_margin,
                    args.surface_granularity,
                )
                right_joints = torch.tensor(rj_np, dtype=torch.float32, device=device)
                left_joints = torch.tensor(lj_np, dtype=torch.float32, device=device)
            else:
                right_joints = (
                    torch.tensor(
                        data.mano_right_joints[t], dtype=torch.float32, device=device
                    )
                    + right_off
                )
                left_joints = (
                    torch.tensor(
                        data.mano_left_joints[t], dtype=torch.float32, device=device
                    )
                    + left_off
                )
            right_joints_wxyz = torch.tensor(
                data.mano_right_joints_wxyz[t], dtype=torch.float32, device=device
            )
            left_joints_wxyz = torch.tensor(
                data.mano_left_joints_wxyz[t], dtype=torch.float32, device=device
            )

            if right_qpos is None:
                right_pos, right_quat_xyzw = wrist_pose_from_mano_joint0(
                    right_joints[0].cpu().numpy(),
                    right_joints_wxyz[0].cpu().numpy(),
                    link_to_site_quat_xyzw=link_to_site_xyzw,
                )
            else:
                right_pos = right_quat_xyzw = None
            if left_qpos is None:
                left_pos, left_quat_xyzw = wrist_pose_from_mano_joint0(
                    left_joints[0].cpu().numpy(),
                    left_joints_wxyz[0].cpu().numpy(),
                    link_to_site_quat_xyzw=link_to_site_xyzw,
                )
            else:
                left_pos = left_quat_xyzw = None

            right_qpos, left_qpos, right_results, left_results = run_frame_ik(
                right_sharpa_kinematics,
                left_sharpa_kinematics,
                right_joints,
                right_joints_wxyz,
                left_joints,
                left_joints_wxyz,
                args.mano_to_robot_scale,
                right_qpos_prev=right_qpos,
                left_qpos_prev=left_qpos,
                right_wrist_position=right_pos,
                right_wrist_quat_xyzw=right_quat_xyzw,
                left_wrist_position=left_pos,
                left_wrist_quat_xyzw=left_quat_xyzw,
            )

            if args.visualize:
                right_sharpa_kinematics.visualize(viser_server, right_qpos)
                left_sharpa_kinematics.visualize(viser_server, left_qpos)
                for obj_idx, obj_name in enumerate(data.object_body_names):
                    if obj_name in viser_object_handles:
                        viser_object_handles[obj_name].position = np.asarray(
                            data.object_body_position[t][obj_idx]
                        )
                        viser_object_handles[obj_name].wxyz = np.asarray(
                            data.object_body_wxyz[t][obj_idx]
                        )
                # Snapshot the solved configs so the post-solve slider can
                # re-pose the robot at any frame without recomputing IK.
                right_qpos_frames.append(np.asarray(right_qpos).copy())
                left_qpos_frames.append(np.asarray(left_qpos).copy())

            robot_right_wrist_position.append(right_results["q"][:3].tolist())
            robot_right_wrist_wxyz.append(
                right_results["q"][3:7][[3, 0, 1, 2]].tolist()
            )
            robot_right_finger_joints.append(right_results["q"][7:].tolist())
            robot_right_frames.append(right_results["frame_pose"].tolist())
            robot_right_frame_task_errors.append(right_results["frame_task_errors"])
            robot_right_num_optimization_iterations.append(
                right_results["num_optimization_iterations"]
            )
            robot_left_wrist_position.append(left_results["q"][:3].tolist())
            robot_left_wrist_wxyz.append(left_results["q"][3:7][[3, 0, 1, 2]].tolist())
            robot_left_finger_joints.append(left_results["q"][7:].tolist())
            robot_left_frames.append(left_results["frame_pose"].tolist())
            robot_left_frame_task_errors.append(left_results["frame_task_errors"])
            robot_left_num_optimization_iterations.append(
                left_results["num_optimization_iterations"]
            )

        if args.save:
            d = data.to_dict()
            d["right_robot_finger_joint_names"] = list(
                right_sharpa_kinematics.robot_finger_joint_names.values()
            )
            d["right_robot_frame_names"] = list(
                right_sharpa_kinematics.robot_frame_names.values()
            )
            d["right_robot_frame_task_names"] = list(
                right_sharpa_kinematics.frame_tasks.keys()
            )
            d["left_robot_finger_joint_names"] = list(
                left_sharpa_kinematics.robot_finger_joint_names.values()
            )
            d["left_robot_frame_names"] = list(
                left_sharpa_kinematics.robot_frame_names.values()
            )
            d["left_robot_frame_task_names"] = list(
                left_sharpa_kinematics.frame_tasks.keys()
            )
            d["robot_right_wrist_position"] = robot_right_wrist_position
            d["robot_right_wrist_wxyz"] = robot_right_wrist_wxyz
            d["robot_right_finger_joints"] = robot_right_finger_joints
            d["robot_right_frames"] = robot_right_frames
            d["robot_right_frame_task_errors"] = robot_right_frame_task_errors
            d["robot_right_num_optimization_iterations"] = (
                robot_right_num_optimization_iterations
            )
            d["robot_left_wrist_position"] = robot_left_wrist_position
            d["robot_left_wrist_wxyz"] = robot_left_wrist_wxyz
            d["robot_left_finger_joints"] = robot_left_finger_joints
            d["robot_left_frames"] = robot_left_frames
            d["robot_left_frame_task_errors"] = robot_left_frame_task_errors
            d["robot_left_num_optimization_iterations"] = (
                robot_left_num_optimization_iterations
            )
            retargeted = ManoSharpaData(**d)
            retargeted.save_to_parquet(
                root_path=str(args.output_dir),
                partition_cols=DEFAULT_PARTITION_COLS,
            )

        # Interactive scrubber: a frame slider + a read-only frame-id readout so
        # the retargeted motion can be inspected frame-by-frame (e.g. to pin
        # motion_start_frame / motion_end_frame). Blocks until Ctrl+C. Mirrors
        # the pattern in scripts/retarget/soma_loader_probe.py.
        if args.visualize:
            num_vis_frames = len(right_qpos_frames)
            frame_slider = viser_server.gui.add_slider(
                "frame", min=0, max=max(num_vis_frames - 1, 0), step=1, initial_value=0
            )
            frame_text = viser_server.gui.add_text(
                "frame_id", initial_value=f"0 / {num_vis_frames - 1}", disabled=True
            )

            def _set_frame(
                t: int,
                *,
                sequence_id: str = sequence_id,
                data: Any = data,
                object_handles: dict[str, Any] = viser_object_handles,
                right_frames: list[np.ndarray] = right_qpos_frames,
                left_frames: list[np.ndarray] = left_qpos_frames,
                num_vis_frames: int = num_vis_frames,
                frame_text: Any = frame_text,
            ) -> None:
                t = int(max(0, min(t, num_vis_frames - 1)))
                right_sharpa_kinematics.visualize(viser_server, right_frames[t])
                left_sharpa_kinematics.visualize(viser_server, left_frames[t])
                for obj_idx, obj_name in enumerate(data.object_body_names):
                    if obj_name in object_handles:
                        object_handles[obj_name].position = np.asarray(
                            data.object_body_position[t][obj_idx]
                        )
                        object_handles[obj_name].wxyz = np.asarray(
                            data.object_body_wxyz[t][obj_idx]
                        )
                frame_text.value = f"{t} / {num_vis_frames - 1}"

            @frame_slider.on_update
            def _on_slider(_: viser.GuiEvent, *, _slider: Any = frame_slider) -> None:
                _set_frame(int(_slider.value))

            _set_frame(0)
            print(
                f"[ego_recon_to_sharpa] viser scrubber for '{sequence_id}': "
                f"{num_vis_frames} frames on port {viser_server.get_port()}. "
                "Drag the 'frame' slider to inspect; Ctrl+C to exit."
            )
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("[ego_recon_to_sharpa] viser stopped.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
