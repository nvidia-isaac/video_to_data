# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retarget the synthetic ``synthbox`` loaded data to Sharpa.

``synthbox`` is a fully synthetic, license-free dataset (see
``scripts/make_synthbox_fixtures.py``) used by the E2E tests in place of TACO.
This retargeter reads the ``synthbox_loaded`` Parquet (``ManoSharpaData`` with
MANO + object fields filled, ``robot_*`` empty), runs per-frame IK to fill the
``robot_*`` fields, and writes ``synthbox_processed``.

It follows the articulated-object convention (bottom + hinged top), so it mirrors
``arctic_to_sharpa.py`` — minus the arctic object-mesh visualization, which the
synthetic box doesn't ship.

Usage:
  python scripts/retarget/synthbox_to_sharpa.py --save
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import (
    ManoSharpaData,
    add_sequence_filter_args,
    filter_sequence_ids,
    list_sequence_ids,
)
from robotic_grounding.retarget.retarget_utils import (
    DEFAULT_PARTITION_COLS,
    run_frame_ik,
    setup_sharpa_kinematics,
    wrist_pose_from_mano_joint0,
)
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Suppress warnings about joint limits being slightly out of bounds
logging.getLogger().setLevel(logging.ERROR)

DEFAULT_INPUT_DIR = HUMAN_MOTION_DATA_DIR / "synthbox" / "synthbox_loaded"
DEFAULT_OUTPUT_DIR = HUMAN_MOTION_DATA_DIR / "synthbox" / "synthbox_processed"

# Rotation offset from MANO link frame to site (wxyz); matches the registry's
# link_to_site_quat_wxyz for synthbox (same convention as arctic).
SYNTHBOX_MANO_LINK_TO_SITE_WXYZ = np.array([0.5, -0.5, 0.5, 0.5])


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retarget synthbox loaded Parquet to Sharpa (run IK, fill robot_*)."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--mano_to_robot_scale", type=float, default=1.0)
    add_sequence_filter_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Read loaded synthbox Parquet, run IK per frame, save retargeted Parquet."""
    device = torch.device(args.device)

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

    link_to_site_xyzw = R.from_quat(
        SYNTHBOX_MANO_LINK_TO_SITE_WXYZ, scalar_first=True
    ).as_quat(
        scalar_first=False
    )  # type: ignore[call-arg]

    for sequence_id in tqdm(sequence_ids):
        data = ManoSharpaData.from_parquet(
            str(args.input_dir),
            filters=[("sequence_id", "=", sequence_id)],
        )
        num_frames = len(data.mano_right_trans)

        # Run IK for each frame and collect robot_* time series
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
            # Same metadata + MANO/object as loaded, plus new robot_* from IK.
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
