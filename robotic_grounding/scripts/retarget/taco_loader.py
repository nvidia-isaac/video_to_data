# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Load TACO dataset into ManoSharpaData schema (MANO + object only, no robot).

Saves Parquet under human_motion_data/taco_loaded/mano_object_only. Use
taco_to_sharpa.py (or a shared retarget script) to retarget to Sharpa.

TACO layout (see vis_taco_data.py):
  dataset_root/
    Object_Poses/{triplet}/{sequence_name}/  -> tool_{name}.npy, target_{name}.npy (4x4)
    Hand_Poses/{triplet}/{sequence_name}/    -> right_hand.pkl, left_hand.pkl, *_hand_shape.pkl
  object_model_root/  -> {name}_cm.obj (mesh in cm; scale 0.01 to m)
"""

import argparse
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.dataset_loader_base import (
    DatasetLoaderBase,
    SequenceInfo,
    load_meshes_to_device,
    poses_to_root_position_and_axis_angle,
)


@dataclass
class TacoSequenceSource:
    """Typed source data for a TACO sequence (stored in SequenceInfo.source)."""

    hand_pose_dir: Path
    object_pose_dir: Path
    num_frames: int
    tool_name: str
    target_name: str
    tool_poses: np.ndarray
    target_poses: np.ndarray


# TACO paths
TACO_DATA_DIR = HUMAN_MOTION_DATA_DIR / "taco"
TACO_OBJECT_MODEL_DIR = TACO_DATA_DIR / "Object_Models" / "object_models_released"
LOADED_SAVE_DIR = HUMAN_MOTION_DATA_DIR / "taco_loaded"

TACO_OBJECT_BODY_NAMES = ["tool", "target"]
TACO_FPS = 30.0


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load TACO sequences into ManoSharpaData schema (MANO + object only)."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=TACO_DATA_DIR,
        help="TACO dataset root (Object_Poses, Hand_Poses).",
    )
    parser.add_argument(
        "--object_model_root",
        type=Path,
        default=TACO_OBJECT_MODEL_DIR,
        help="Directory with {name}_cm.obj meshes.",
    )
    parser.add_argument(
        "--triplet",
        type=str,
        default=None,
        help='Tool-action-object triplet, e.g. "(brush, brush, bowl)". If not set, process all triplets.',
    )
    parser.add_argument(
        "--sample_triplets",
        type=int,
        default=None,
        help="Randomly sample N triplets from the dataset. Only used when --triplet is not set.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=LOADED_SAVE_DIR,
        help="Parent directory for Parquet; data written to <output_dir>.",
    )
    return parser.parse_args()


def _list_taco_triplets(dataset_root: Path) -> list[str]:
    """List all triplet directory names under Hand_Poses/."""
    hand_dir = dataset_root / "Hand_Poses"
    if not hand_dir.is_dir():
        return []
    return sorted([d.name for d in hand_dir.iterdir() if d.is_dir()])


def _list_taco_sequences(dataset_root: Path, triplet: str) -> list[str]:
    """List sequence names under Hand_Poses/{triplet}/."""
    hand_dir = dataset_root / "Hand_Poses" / triplet
    if not hand_dir.is_dir():
        return []
    return sorted([d.name for d in hand_dir.iterdir() if d.is_dir()])


def _discover_tool_and_target(object_pose_dir: Path) -> tuple[str, str]:
    """Get tool_name and target_name from Object_Poses dir."""
    tool_name, target_name = None, None
    for f in object_pose_dir.iterdir():
        if not f.is_file():
            continue
        name = f.stem
        if name.startswith("tool_"):
            tool_name = name.replace("tool_", "", 1)
        elif name.startswith("target_"):
            target_name = name.replace("target_", "", 1)
    if tool_name is None or target_name is None:
        raise FileNotFoundError(
            f"Could not find tool_*.npy and target_*.npy in {object_pose_dir}"
        )
    return tool_name, target_name


def _load_taco_object_poses(
    object_pose_dir: Path,
    tool_name: str,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tool and target poses; each (N, 4, 4) in world/meter frame."""
    tool_path = object_pose_dir / f"tool_{tool_name}.npy"
    target_path = object_pose_dir / f"target_{target_name}.npy"
    if not tool_path.exists():
        raise FileNotFoundError(f"Tool poses not found: {tool_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target poses not found: {target_path}")
    tool_poses = np.load(tool_path)
    target_poses = np.load(target_path)
    if tool_poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (N, 4, 4), got {tool_poses.shape}")
    N = tool_poses.shape[0]
    if target_poses.shape[0] != N:
        raise ValueError(f"Tool has {N} frames, target has {target_poses.shape[0]}")
    return tool_poses, target_poses


def load_taco_hand_shape(hand_pose_dir: Path, side: str) -> np.ndarray:
    """Load hand shape beta (10,) from *_hand_shape.pkl."""
    path = hand_pose_dir / f"{side}_hand_shape.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Hand shape not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    beta = data["hand_shape"]
    if hasattr(beta, "detach"):
        beta = beta.detach().cpu().numpy()
    return np.asarray(beta, dtype=np.float32).reshape(10)


def _mano_params_from_pkl(
    hand_pose_dir: Path,
    side: str,
    N: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load integrated hand pkl: dict of frames with 'hand_pose' (48) and 'hand_trans' (3)."""
    path = hand_pose_dir / f"{side}_hand.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    keys = sorted(data.keys())
    theta_list = []
    trans_list = []
    for key in keys:
        frame_data = data[key]
        hp = frame_data["hand_pose"]
        ht = frame_data["hand_trans"]
        if hasattr(hp, "detach"):
            hp = hp.detach().cpu().numpy()
        if hasattr(ht, "detach"):
            ht = ht.detach().cpu().numpy()
        theta_list.append(np.asarray(hp, dtype=np.float32).reshape(48))
        trans_list.append(np.asarray(ht, dtype=np.float32).reshape(3))
    if not theta_list:
        return None
    batch_theta = np.stack(theta_list, axis=0)
    batch_trans = np.stack(trans_list, axis=0)
    global_orient = batch_theta[:, :3]
    finger_pose = batch_theta[:, 3:48]
    if batch_theta.shape[0] != N:
        raise ValueError(
            f"Hand pkl has {batch_theta.shape[0]} frames, object poses have {N}"
        )
    return (
        torch.from_numpy(batch_trans).float().to(device),
        torch.from_numpy(global_orient).float().to(device),
        torch.from_numpy(finger_pose).float().to(device),
    )


def _triplet_to_safe_id(triplet: str) -> str:
    """Normalize triplet string for use in sequence_id."""
    return re.sub(r"[\s(),]", "_", triplet).strip("_")


class TacoDatasetLoader(DatasetLoaderBase):
    """TACO dataset loader."""

    def _list_sequences_for_triplet(
        self, dataset_root: Path, triplet: str
    ) -> list[SequenceInfo]:
        """List TACO sequences for a single triplet."""
        sequences = _list_taco_sequences(dataset_root, triplet)
        if not sequences:
            return []
        out: list[SequenceInfo] = []
        for sequence in sequences:
            object_pose_dir = dataset_root / "Object_Poses" / triplet / sequence
            hand_pose_dir = dataset_root / "Hand_Poses" / triplet / sequence
            if not object_pose_dir.is_dir() or not hand_pose_dir.is_dir():
                continue
            try:
                tool_name, target_name = _discover_tool_and_target(object_pose_dir)
                tool_poses, target_poses = _load_taco_object_poses(
                    object_pose_dir, tool_name, target_name
                )
            except (FileNotFoundError, ValueError):
                continue
            N = tool_poses.shape[0]
            sequence_id = f"taco_{_triplet_to_safe_id(triplet)}_{sequence}"
            raw_motion_file = f"{triplet}/{sequence}"
            object_name = f"{tool_name}_{target_name}"
            out.append(
                SequenceInfo(
                    sequence_id=sequence_id,
                    raw_motion_file=raw_motion_file,
                    object_name=object_name,
                    object_body_names=TACO_OBJECT_BODY_NAMES,
                    source=TacoSequenceSource(
                        hand_pose_dir=hand_pose_dir,
                        object_pose_dir=object_pose_dir,
                        num_frames=N,
                        tool_name=tool_name,
                        target_name=target_name,
                        tool_poses=tool_poses,
                        target_poses=target_poses,
                    ),
                )
            )
        return out

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """List TACO sequences for the given dataset root and triplet(s).

        If args.triplet is None, all triplets under Hand_Poses/ are processed.
        """
        dataset_root = Path(args.dataset_root)
        if args.triplet is not None:
            return self._list_sequences_for_triplet(dataset_root, args.triplet)
        triplets = _list_taco_triplets(dataset_root)
        sample_n = getattr(args, "sample_triplets", None)
        if sample_n is not None and sample_n < len(triplets):
            triplets = sorted(random.sample(triplets, sample_n))
            print(f"Sampled {sample_n} triplets: {triplets}")
        out: list[SequenceInfo] = []
        for triplet in triplets:
            out.extend(self._list_sequences_for_triplet(dataset_root, triplet))
        return out

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO parameters from TACO hand pkl files for the sequence."""
        src: TacoSequenceSource = sequence_info.source
        N = src.num_frames
        right_params = _mano_params_from_pkl(src.hand_pose_dir, "right", N, device)
        left_params = _mano_params_from_pkl(src.hand_pose_dir, "left", N, device)
        if right_params is None or left_params is None:
            raise FileNotFoundError("Missing hand pkl data")
        right_transl, right_global_orient, right_finger_pose = right_params
        left_transl, left_global_orient, left_finger_pose = left_params
        right_betas = (
            torch.from_numpy(load_taco_hand_shape(src.hand_pose_dir, "right"))
            .float()
            .to(device)
        )
        left_betas = (
            torch.from_numpy(load_taco_hand_shape(src.hand_pose_dir, "left"))
            .float()
            .to(device)
        )
        H = N
        return {
            "right_global_orient": right_global_orient,
            "right_finger_pose": right_finger_pose,
            "right_trans": right_transl,
            "right_betas": right_betas,
            "right_fitting_err": torch.zeros(H, device=device),
            "left_global_orient": left_global_orient,
            "left_finger_pose": left_finger_pose,
            "left_trans": left_transl,
            "left_betas": left_betas,
            "left_fitting_err": torch.zeros(H, device=device),
            "H": H,
        }

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Load object data: name -> (pose, root_position, root_axis_angle, articulation)."""
        src: TacoSequenceSource = sequence_info.source
        tool_root_pos, tool_root_aa = poses_to_root_position_and_axis_angle(
            src.tool_poses
        )
        target_root_pos, target_root_aa = poses_to_root_position_and_axis_angle(
            src.target_poses
        )
        return {
            TACO_OBJECT_BODY_NAMES[0]: (
                src.tool_poses,
                tool_root_pos,
                tool_root_aa,
                None,
            ),
            TACO_OBJECT_BODY_NAMES[1]: (
                src.target_poses,
                target_root_pos,
                target_root_aa,
                None,
            ),
        }

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple[
        dict[str, Any],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        bool,
    ]:
        """Load TACO object meshes (tool/target _cm.obj) for the sequence."""
        src: TacoSequenceSource = sequence_info.source
        object_model_root = Path(
            getattr(self._args, "object_model_root", TACO_OBJECT_MODEL_DIR)
        )
        mesh_paths = {
            "tool": str(object_model_root / f"{src.tool_name}_cm.obj"),
            "target": str(object_model_root / f"{src.target_name}_cm.obj"),
        }
        return load_meshes_to_device(mesh_paths, device, vertex_scale=0.01)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """Return MANO model kwargs for TACO (flat_hand_mean=True, center_idx=0)."""
        return {"flat_hand_mean": True, "center_idx": 0}

    def get_fps(self) -> float:
        """Return TACO sequence FPS."""
        return TACO_FPS

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return paths to TACO object meshes (tool and target _cm.obj)."""
        src: TacoSequenceSource = sequence_info.source
        object_model_root = Path(
            getattr(self._args, "object_model_root", TACO_OBJECT_MODEL_DIR)
        )
        return [
            str(object_model_root / f"{src.tool_name}_cm.obj"),
            str(object_model_root / f"{src.target_name}_cm.obj"),
        ]


def main(args: argparse.Namespace) -> None:
    """Load TACO sequences and save as ManoSharpaData (MANO + object only)."""
    loader = TacoDatasetLoader()
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
