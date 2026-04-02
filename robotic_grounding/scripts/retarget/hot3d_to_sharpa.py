# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Retarget Hot3D loaded data (ManoSharpaData with MANO+object only) to Sharpa.

Reads Parquet from hot3d_loader.py output (human_motion_data/hot3d_loaded),
runs IK per frame to fill robot_* fields, and saves to hot3d_processed.

Usage:
  1. python scripts/retarget/hot3d_loader.py --save
  2. python scripts/retarget/hot3d_to_sharpa.py --save
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import viser
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData, list_sequence_ids
from robotic_grounding.retarget.retarget_utils import (
    DEFAULT_PARTITION_COLS,
    run_frame_ik,
    setup_sharpa_kinematics,
    wrist_pose_from_mano_joint0,
)
from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)

DEFAULT_INPUT_DIR = HUMAN_MOTION_DATA_DIR / "hot3d_loaded"
DEFAULT_OUTPUT_DIR = HUMAN_MOTION_DATA_DIR / "hot3d_processed"

# Hot3D has no wrist link-to-site rotation offset (same as OakInk2).
HOT3D_LINK_TO_SITE_QUAT_XYZW = None


def _load_object_viser_handles(
    viser_server: viser.ViserServer,
    object_mesh_paths: list[str],
    object_body_names: list[str],
) -> dict[str, Any]:
    """Load Hot3D object meshes from stored schema paths and add to viser scene.

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
    """Parse command-line arguments for the Hot3D-to-Sharpa retargeting script."""
    parser = argparse.ArgumentParser(
        description="Retarget Hot3D loaded Parquet data to Sharpa (run IK, fill robot_*)."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--mano_to_robot_scale", type=float, default=1.2)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Read loaded Hot3D Parquet, run IK per frame, save retargeted Parquet."""
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
    print(f"Found {len(sequence_ids)} sequences in {args.input_dir}")

    link_to_site_xyzw = HOT3D_LINK_TO_SITE_QUAT_XYZW

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

        for t in range(num_frames):
            right_joints = torch.tensor(
                data.mano_right_joints[t], dtype=torch.float32, device=device
            )
            right_joints_wxyz = torch.tensor(
                data.mano_right_joints_wxyz[t], dtype=torch.float32, device=device
            )
            left_joints = torch.tensor(
                data.mano_left_joints[t], dtype=torch.float32, device=device
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
