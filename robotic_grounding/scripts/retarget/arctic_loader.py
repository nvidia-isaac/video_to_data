# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load ARCTIC dataset into ManoSharpaData schema (MANO + object only, no robot).

Saves Parquet under human_motion_data/arctic_loaded/mano_object_only. Use
arctic_to_sharpa.py to retarget the saved data to Sharpa robot and write
arctic_processed.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from robotic_grounding.retarget import (
    ASSETS_DIR,
    HUMAN_MOTION_DATA_DIR,
)
from robotic_grounding.retarget.dataset_loader_base import (
    DatasetLoaderBase,
    SequenceInfo,
    load_meshes_to_device,
)
from scipy.spatial.transform import Rotation as R

# Suppress warnings about joint limits being slightly out of bounds
logging.getLogger().setLevel(logging.ERROR)

# ARCTIC paths
ARCTIC_MOTION_DIR = HUMAN_MOTION_DATA_DIR / "arctic"
ARCTIC_URDF_DIR = ASSETS_DIR / "urdfs" / "arctic"
ARCTIC_MESH_DIR = ASSETS_DIR / "meshes" / "arctic"
LOADED_SAVE_DIR = HUMAN_MOTION_DATA_DIR / "arctic_loaded"

OBJECT_BODY_NAMES = ["bottom", "top"]
FRAME_START = 0
FRAME_END_OFFSET = 0
ARCTIC_FPS = 30.0

ARCTIC_MANO_KWARGS = {"flat_hand_mean": False, "center_idx": None}


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load ARCTIC sequences into ManoSharpaData schema (MANO + object only)."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=LOADED_SAVE_DIR,
        help="Parent directory for Parquet output; data is written to <output_dir>.",
    )
    return parser.parse_args()


def load_arctic_mano_data(mano_data_file: Path, device: torch.device) -> dict[str, Any]:
    """Load MANO parameters from an ARCTIC .mano.npy file."""
    mano_data = np.load(mano_data_file, allow_pickle=True).item()
    return {
        "right_global_orient": torch.from_numpy(mano_data["right"]["rot"]).to(device),
        "right_finger_pose": torch.from_numpy(mano_data["right"]["pose"]).to(device),
        "right_trans": torch.from_numpy(mano_data["right"]["trans"]).to(device),
        "right_betas": torch.from_numpy(mano_data["right"]["shape"]).to(device),
        "right_fitting_err": torch.tensor(mano_data["right"]["fitting_err"]).to(device),
        "left_global_orient": torch.from_numpy(mano_data["left"]["rot"]).to(device),
        "left_finger_pose": torch.from_numpy(mano_data["left"]["pose"]).to(device),
        "left_trans": torch.from_numpy(mano_data["left"]["trans"]).to(device),
        "left_betas": torch.from_numpy(mano_data["left"]["shape"]).to(device),
        "left_fitting_err": torch.tensor(mano_data["left"]["fitting_err"]).to(device),
        "H": len(mano_data["right"]["rot"]),
    }


def load_arctic_object_data(
    mano_data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load object pose and articulation from ARCTIC .object.npy."""
    object_path = str(mano_data_path).replace(".mano.", ".object.")
    object_data = np.load(object_path, allow_pickle=True)
    object_articulation = object_data[:, 0]
    object_axis_angle = object_data[:, 1:4]
    object_translation = object_data[:, 4:] / 1000.0
    return object_articulation, object_axis_angle, object_translation


def setup_arctic_mujoco_object(
    object_name: str,
) -> tuple[mujoco.MjModel, mujoco.MjData, int, int]:
    """Load MuJoCo model for ARCTIC object; return body/joint ids for FK."""
    object_urdf_path = ARCTIC_URDF_DIR / f"{object_name}.urdf"
    model = mujoco.MjModel.from_xml_path(str(object_urdf_path))
    data = mujoco.MjData(model)
    top_body_id = model.body("top").id
    rotation_joint_id = model.joint("rotation").id
    return model, data, top_body_id, rotation_joint_id


def get_arctic_object_world_transforms(
    frame_id: int,
    object_articulation: np.ndarray,
    object_axis_angle: np.ndarray,
    object_translation: np.ndarray,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    top_body_id: int,
    rotation_joint_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute world position and quat (wxyz) for bottom and top at one frame."""
    world_p_bottom = object_translation[frame_id]
    world_q_bottom = R.from_rotvec(object_axis_angle[frame_id]).as_quat(
        scalar_first=True
    )
    world_t_bottom = np.eye(4)
    world_t_bottom[:3, :3] = R.from_quat(world_q_bottom, scalar_first=True).as_matrix()
    world_t_bottom[:3, 3] = world_p_bottom

    mj_data.qpos[rotation_joint_id] = object_articulation[frame_id]
    mujoco.mj_forward(mj_model, mj_data)
    bottom_t_top = np.eye(4)
    bottom_t_top[:3, :3] = np.array(mj_data.xmat[top_body_id]).reshape(3, 3)
    bottom_t_top[:3, 3] = mj_data.xpos[top_body_id]
    world_t_top = world_t_bottom @ bottom_t_top
    world_p_top = world_t_top[:3, 3]
    world_q_top = R.from_matrix(world_t_top[:3, :3]).as_quat(scalar_first=True)

    world_p_objects = np.vstack([world_p_bottom, world_p_top])
    world_q_objects = np.vstack([world_q_bottom, world_q_top])
    return (
        world_p_bottom,
        world_q_bottom,
        world_p_top,
        world_q_top,
        world_p_objects,
        world_q_objects,
    )


def _arctic_world_poses_one_frame(
    frame_id: int,
    object_articulation: np.ndarray,
    object_axis_angle: np.ndarray,
    object_translation: np.ndarray,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    top_body_id: int,
    rotation_joint_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (world_t_bottom, world_t_top) 4x4 for one frame."""
    world_t_bottom = np.eye(4)
    world_q_bottom = R.from_rotvec(object_axis_angle[frame_id]).as_quat(
        scalar_first=True
    )
    world_t_bottom[:3, :3] = R.from_quat(world_q_bottom, scalar_first=True).as_matrix()
    world_t_bottom[:3, 3] = object_translation[frame_id]
    mj_data.qpos[rotation_joint_id] = object_articulation[frame_id]
    mujoco.mj_forward(mj_model, mj_data)
    bottom_t_top = np.eye(4)
    bottom_t_top[:3, :3] = np.array(mj_data.xmat[top_body_id]).reshape(3, 3)
    bottom_t_top[:3, 3] = mj_data.xpos[top_body_id]
    world_t_top = world_t_bottom @ bottom_t_top
    return world_t_bottom, world_t_top


class ArcticDatasetLoader(DatasetLoaderBase):
    """ARCTIC dataset loader."""

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """List ARCTIC sequences (discover *.mano.npy, exclude scissor)."""
        mano_files = sorted(
            [
                f
                for f in ARCTIC_MOTION_DIR.glob("*/*.mano.npy")
                if "scissor" not in f.name
            ]
        )
        out = []
        for mano_data_file in mano_files:
            object_name = mano_data_file.name.split("_")[0]
            raw_motion_file = str(Path(*mano_data_file.parts[-3:]))[:-9]
            sequence_id = raw_motion_file.replace("/", "_")
            out.append(
                SequenceInfo(
                    sequence_id=sequence_id,
                    raw_motion_file=raw_motion_file,
                    object_name=object_name,
                    object_body_names=OBJECT_BODY_NAMES,
                    source=mano_data_file,
                )
            )
        return out

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO parameters from ARCTIC .mano.npy for the sequence."""
        mano_data_file = sequence_info.source
        if mano_data_file is None:
            raise FileNotFoundError("ARCTIC sequence has no source path")
        return load_arctic_mano_data(Path(mano_data_file), device)

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Load object data: name -> (pose, root_position, root_axis_angle, articulation)."""
        mano_data_file = sequence_info.source
        if mano_data_file is None:
            raise FileNotFoundError("ARCTIC sequence has no source path")
        art, axis, trans = load_arctic_object_data(str(mano_data_file))
        mj_model, mj_data, top_id, rot_id = setup_arctic_mujoco_object(
            sequence_info.object_name
        )
        n_frames = len(art)
        bottom_poses = np.zeros((n_frames, 4, 4), dtype=np.float64)
        top_poses = np.zeros((n_frames, 4, 4), dtype=np.float64)
        for i in range(n_frames):
            world_t_bottom, world_t_top = _arctic_world_poses_one_frame(
                i, art, axis, trans, mj_model, mj_data, top_id, rot_id
            )
            bottom_poses[i] = world_t_bottom
            top_poses[i] = world_t_top
        # Root (bottom) has articulation; top uses same root_position/root_axis_angle
        return {
            OBJECT_BODY_NAMES[0]: (bottom_poses, trans, axis, art),
            OBJECT_BODY_NAMES[1]: (top_poses, trans, axis, None),
        }

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple[
        dict[str, Any],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        bool,
    ]:
        """Load ARCTIC object part meshes (bottom/top) for the sequence object."""
        mesh_paths = {
            part: str(
                ARCTIC_MESH_DIR
                / sequence_info.object_name
                / f"{part}_watertight_tiny.obj"
            )
            for part in sequence_info.object_body_names
        }
        return load_meshes_to_device(mesh_paths, device)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """Return MANO model kwargs for ARCTIC (flat_hand_mean=False)."""
        return ARCTIC_MANO_KWARGS

    def get_fps(self) -> float:
        """Return ARCTIC sequence FPS."""
        return ARCTIC_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return paths to ARCTIC object part meshes (bottom/top watertight_tiny.obj)."""
        return [
            str(
                ARCTIC_MESH_DIR
                / sequence_info.object_name
                / f"{part}_watertight_tiny.obj"
            )
            for part in sequence_info.object_body_names
        ]

    def get_frame_range(self, num_frames: int) -> tuple[int, int]:
        """Return (start, end) frame indices; ARCTIC trims first/last frames."""
        return FRAME_START, num_frames - FRAME_END_OFFSET


def main(args: argparse.Namespace) -> None:
    """Load ARCTIC sequences and save as ManoSharpaData (MANO + object only)."""
    loader = ArcticDatasetLoader()
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
