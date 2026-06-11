# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for loading hand-object datasets into ManoSharpaData schema.

Subclass and implement the abstract methods to add a new dataset (e.g. ARCTIC, TACO).
"""

import argparse
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# FK / contact code ported into this module (manotorch-tainted or FK-output utils).
from v2d.task_library_loader.lib.contact_utils import (
    approximate_contact_with_id,
    compute_hand_link_contact_positions,
    find_object_contact_positions,
)
from v2d.task_library_loader.lib.distance_utils import (
    compute_tip_to_object_surface_distance,
)
from v2d.task_library_loader.lib.read_mano import MANO

# Clean shared code imported from robotic_grounding (installed dependency; GPL-free).
from robotic_grounding.retarget.data_logger import ManoSharpaData, shard_matches
from robotic_grounding.retarget.naming import make_usd_safe
from robotic_grounding.retarget.params import MANO_HAND_LINKS, NUM_MANO_LINKS

# Partition columns for the output Parquet. Copied from robotic_grounding's
# retarget_utils to avoid importing it here (it would pull pink/pinocchio IK deps).
DEFAULT_PARTITION_COLS = ["sequence_id", "robot_name"]
# ``make_usd_safe`` (imported above from naming.py) is re-exported here so the
# ported loaders' ``from ...dataset_loader_base import make_usd_safe`` keep working.


@dataclass
class SequenceInfo:
    """Metadata for one sequence to process."""

    sequence_id: str
    raw_motion_file: str
    object_name: str
    object_body_names: list[str]
    source: Any = None  # Subclass-specific (e.g. Path to .mano.npy, or (hand_dir, N))


@dataclass
class FrameObjectPoses:
    """Per-frame object poses and optional combined mesh for tips-distance."""

    object_body_position: list[list[float]]
    object_body_wxyz: list[list[float]]
    object_root_axis_angle: list[float]
    object_root_position: list[float]
    object_articulation: float
    object_surface_points_world: torch.Tensor
    object_surface_normals_world: torch.Tensor
    object_surface_points_part_ids: torch.Tensor


def poses_to_root_position_and_axis_angle(
    poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract root position and axis-angle from (N, 4, 4) world transform matrices.

    Returns:
        root_position: (N, 3) translation column.
        root_axis_angle: (N, 3) axis-angle from rotation matrices.
    """
    root_position = poses[:, :3, 3]
    root_axis_angle = np.array(
        [R.from_matrix(poses[i, :3, :3]).as_rotvec() for i in range(len(poses))]
    )
    return root_position, root_axis_angle


def load_meshes_to_device(
    mesh_paths: dict[str, str],
    device: torch.device,
    vertex_scale: float = 1.0,
    num_surface_points: int = 4096,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    bool,
]:
    """Load trimesh objects by part name, convert verts/faces to device tensors.

    Args:
        mesh_paths: Mapping of part name to mesh file path.
        device: Target device for tensors.
        vertex_scale: Factor to scale vertex positions (e.g. 0.01 for cm->m).
        num_surface_points: Number of surface points to sample per part.

    Returns:
        (meshes, verts, faces, compute_tips_dist) -- meshes is the raw trimesh
        per part; verts/faces are on device; compute_tips_dist is False if any
        part mesh was not found.
    """
    meshes: dict[str, Any] = {}
    verts: dict[str, torch.Tensor] = {}
    faces: dict[str, torch.Tensor] = {}
    surface_points: dict[str, torch.Tensor] = {}
    surface_normals: dict[str, torch.Tensor] = {}
    compute_tips_dist = True

    for part, path_str in mesh_paths.items():
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Mesh not found at {path}, skipping {part}")
            compute_tips_dist = False
            continue
        mesh = trimesh.load(str(path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if vertex_scale != 1.0:
            mesh.vertices *= vertex_scale
        meshes[part] = mesh
        verts[part] = torch.from_numpy(mesh.vertices).float().to(device)
        faces[part] = torch.from_numpy(mesh.faces).long().to(device)

        pts, face_idx = trimesh.sample.sample_surface_even(mesh, num_surface_points)
        surface_points[part] = torch.from_numpy(pts).float().to(device)
        part_normals = (
            torch.from_numpy(mesh.face_normals[face_idx]).float().to(device)
        )  # point outward
        part_normals = part_normals / torch.norm(
            part_normals, dim=-1, keepdim=True
        ).clamp(min=1e-6)
        surface_normals[part] = -part_normals  # point inward

    return meshes, verts, faces, surface_points, surface_normals, compute_tips_dist


def build_combined_object_surface(
    object_surface_points: dict[str, torch.Tensor],
    object_surface_normals: dict[str, torch.Tensor],
    body_names: list[str],
    world_transforms: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine per-body surface points and normals transformed to world frame.

    Args:
        object_surface_points: Surface points per body name (on device).
        object_surface_normals: Surface normals per body name (on device).
        body_names: Order of bodies (must match world_transforms).
        world_transforms: List of 4x4 world matrices, one per body in body_names order.
        device: Target device for output tensors.

    Returns:
        combined_verts: (V, 3) all vertices in world frame.
        combined_faces: (F, 3) face indices with offset for concatenated verts.
    """
    all_verts: list[torch.Tensor] = []
    all_normals: list[torch.Tensor] = []
    all_part_ids: list[torch.Tensor] = []
    for body_idx, (name, T) in enumerate(
        zip(body_names, world_transforms, strict=True)
    ):
        if name not in object_surface_points or name not in object_surface_normals:
            continue
        R_mat = torch.from_numpy(np.asarray(T)[:3, :3]).float().to(device)
        t_vec = torch.from_numpy(np.asarray(T)[:3, 3]).float().to(device)
        verts = (R_mat @ object_surface_points[name].T).T + t_vec
        normals = (R_mat @ object_surface_normals[name].T).T
        part_ids = torch.full((verts.shape[0],), body_idx + 1, device=device)
        all_verts.append(verts)
        all_normals.append(normals)
        all_part_ids.append(part_ids)
    if not all_verts:
        raise ValueError("No mesh parts found to combine")
    return (
        torch.cat(all_verts, dim=0),
        torch.cat(all_normals, dim=0),
        torch.cat(all_part_ids, dim=0),
    )


class DatasetLoaderBase(ABC):
    """Base class for loading a dataset into ManoSharpaData (MANO + object only).

    Subclasses implement: list_sequences, load_mano_data, load_object_data,
    load_object_meshes, get_mano_kwargs, get_fps. Optionally override
    get_frame_object_poses, get_object_mesh_paths, get_frame_range.
    """

    @abstractmethod
    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """List sequences to process. Uses args for dataset paths/filters."""
        ...

    @abstractmethod
    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Load MANO parameters for the sequence.

        Returns dict with: H, right_global_orient, right_finger_pose, right_trans,
        right_betas, right_fitting_err, left_global_orient, left_finger_pose,
        left_trans, left_betas, left_fitting_err. All tensors on device.
        """
        ...

    @abstractmethod
    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Load object pose data (passed to get_frame_object_poses).

        Return a dictionary mapping each object/body name to a tuple
        (pose, root_position, root_axis_angle, articulation):
        - pose: (N, 4, 4) world transform per frame
        - root_position: (N, 3) object root position per frame
        - root_axis_angle: (N, 3) object root axis-angle per frame
        - articulation: (N,) array or None if no articulation
        """
        ...

    @abstractmethod
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
        """Load object part meshes.

        Returns:
            object_mesh_meshes: trimesh (or Scene) per part, for visualization.
            object_mesh_verts: vertices per part on device.
            object_mesh_faces: faces per part on device.
            object_surface_points: surface points per part.
            object_surface_normals: surface normals per part.
            compute_tips_dist: whether tips-distance can be computed.
        """
        ...

    def get_frame_object_poses(
        self,
        frame_id: int,
        object_data: dict[str, Any],
        object_surface_points: dict[str, torch.Tensor],
        object_surface_normals: dict[str, torch.Tensor],
        device: torch.device,
    ) -> FrameObjectPoses:
        """Compute per-body world poses and optional combined mesh for one frame.

        Uses the (pose, root_position, root_axis_angle, articulation) tuple
        convention from load_object_data(). Override if custom logic is needed.
        """
        body_names = list(object_data.keys())
        world_transforms = [object_data[name][0][frame_id] for name in body_names]

        object_body_position = [t[:3, 3].tolist() for t in world_transforms]
        object_body_wxyz = [
            R.from_matrix(t[:3, :3]).as_quat(scalar_first=True).tolist()
            for t in world_transforms
        ]

        _pose, root_position, root_axis_angle, articulation = object_data[body_names[0]]

        (
            object_surface_points_world,
            object_surface_normals_world,
            object_surface_points_part_ids,
        ) = build_combined_object_surface(
            object_surface_points,
            object_surface_normals,
            body_names,
            world_transforms,
            device,
        )

        return FrameObjectPoses(
            object_body_position=object_body_position,
            object_body_wxyz=object_body_wxyz,
            object_root_axis_angle=root_axis_angle[frame_id].tolist(),
            object_root_position=root_position[frame_id].tolist(),
            object_articulation=(
                float(articulation[frame_id]) if articulation is not None else 0.0
            ),
            object_surface_points_world=object_surface_points_world,
            object_surface_normals_world=object_surface_normals_world,
            object_surface_points_part_ids=object_surface_points_part_ids,
        )

    @abstractmethod
    def get_mano_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for MANO(...), e.g. flat_hand_mean, center_idx."""
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """Frames per second for this dataset."""
        ...

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return paths to object mesh files (one per object body, same order as object_body_names). Override in subclass."""
        return []

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Return paths to object URDF files (one per object body, same order as object_body_names). Override in subclass."""
        return []

    def get_frame_range(self, num_frames: int) -> tuple[int, int]:
        """Return (start, end) frame indices to process. Override to trim frames."""
        return 0, num_frames

    @staticmethod
    def add_common_args(
        parser: argparse.ArgumentParser,
        *,
        dataset_root: Any,
        object_model_root: Any,
        mesh_dir: Any,
        output_dir: Any,
    ) -> None:
        """Register the args-first contract shared by every loader.

        Every loader resolves its raw data (``--dataset_root``), object URDFs
        (``--object_model_root``), object meshes (``--mesh_dir``) and MANO models
        (``--mano_model_dir``) from these flags. The OSMO load workflow passes them
        as swift-fetched runtime paths; the per-dataset values passed here are
        local-dev fallbacks. Also registers ``--output_dir``/``--device``/
        ``--visualize``/``--save`` and the sequence-filter args (via
        :meth:`add_filter_args`).
        """
        parser.add_argument(
            "--dataset_root",
            type=Path,
            default=dataset_root,
            help="Raw dataset root for this dataset.",
        )
        parser.add_argument(
            "--object_model_root",
            type=Path,
            default=object_model_root,
            help="Directory with this dataset's per-object rigid URDFs.",
        )
        parser.add_argument(
            "--mesh_dir",
            type=Path,
            default=mesh_dir,
            help="Directory with this dataset's object meshes.",
        )
        parser.add_argument(
            "--output_dir",
            type=Path,
            default=output_dir,
            help="Parent directory for Parquet output.",
        )
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--visualize", action="store_true", default=False)
        parser.add_argument("--save", action="store_true", default=False)
        DatasetLoaderBase.add_filter_args(parser)

    @staticmethod
    def add_filter_args(parser: argparse.ArgumentParser) -> None:
        """Add common sequence filtering args. Call from each loader's parse_args()."""
        parser.add_argument(
            "--mano_model_dir",
            type=str,
            default=None,
            help=(
                "Directory containing the MANO models (a models/ subdir with "
                "MANO_RIGHT.pkl / MANO_LEFT.pkl). Required to run MANO FK; "
                "supplied at runtime and never vendored."
            ),
        )
        group = parser.add_argument_group("sequence filtering")
        group.add_argument(
            "--sequence_id",
            type=str,
            default=None,
            help="Process a single sequence by exact ID.",
        )
        group.add_argument(
            "--sequence_pattern",
            type=str,
            default=None,
            help="Regex pattern to filter sequence IDs (e.g., '.*box.*').",
        )
        group.add_argument(
            "--sequence_file",
            type=str,
            default=None,
            help="Text file with sequence IDs to process (one per line).",
        )
        group.add_argument(
            "--max_sequences",
            type=int,
            default=None,
            help="Limit to first N sequences after filtering.",
        )
        group.add_argument(
            "--list_only",
            action="store_true",
            default=False,
            help="List matching sequence IDs and exit without processing.",
        )
        group.add_argument(
            "--shard_id",
            type=int,
            default=0,
            help="Shard index (0-based) for parallel processing.",
        )
        group.add_argument(
            "--num_shards",
            type=int,
            default=1,
            help="Total number of shards.  1 = no sharding (default).",
        )

    @staticmethod
    def _apply_sequence_filters(
        sequences: list["SequenceInfo"], args: Any
    ) -> list["SequenceInfo"]:
        """Apply common sequence filters (incl. shard partitioning) to the list."""
        if getattr(args, "sequence_id", None):
            sequences = [s for s in sequences if s.sequence_id == args.sequence_id]
        if getattr(args, "sequence_pattern", None):
            pat = re.compile(args.sequence_pattern)
            sequences = [s for s in sequences if pat.search(s.sequence_id)]
        if getattr(args, "sequence_file", None):
            with open(args.sequence_file) as f:
                ids = {line.strip() for line in f if line.strip()}
            sequences = [s for s in sequences if s.sequence_id in ids]
        if getattr(args, "max_sequences", None):
            sequences = sequences[: args.max_sequences]

        num_shards = getattr(args, "num_shards", 1) or 1
        shard_id = getattr(args, "shard_id", 0) or 0
        if num_shards > 1:
            sequences = [
                s
                for s in sequences
                if shard_matches(s.sequence_id, shard_id, num_shards)
            ]
        return sequences

    def run(self, args: Any) -> None:
        """Common pipeline: list sequences, load MANO/object, log timesteps, save."""
        self._args = args
        device = torch.device(args.device)

        if args.visualize:
            viser_server = viser.ViserServer()
        else:
            viser_server = None

        if args.save:
            args.output_dir.mkdir(parents=True, exist_ok=True)

        sequences = self.list_sequences(args)
        sequences = self._apply_sequence_filters(sequences, args)

        if getattr(args, "list_only", False):
            for s in sequences:
                print(s.sequence_id)
            return

        print(f"Found {len(sequences)} sequences")

        if getattr(args, "mano_model_dir", None) is None:
            raise ValueError(
                "--mano_model_dir is required: pass the directory containing the "
                "MANO models (models/MANO_RIGHT.pkl, MANO_LEFT.pkl)."
            )
        mano_kwargs = self.get_mano_kwargs()
        mano = MANO(
            mano_assets_root=args.mano_model_dir,
            gender="neutral",
            device=device,
            **mano_kwargs,
        )
        viser_object_handles: dict[str, Any] = {}
        viser_contact_handles: list[Any] = []

        for sequence_info in tqdm(sequences):
            try:
                raw_data = self.load_mano_data(sequence_info, device)
            except (
                FileNotFoundError,
                ValueError,
                KeyError,
                pickle.UnpicklingError,
            ) as e:
                print(f"Skipping {sequence_info.sequence_id}: {e}")
                continue

            # Number of frames
            H = raw_data["H"]
            try:
                object_data = self.load_object_data(sequence_info)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {sequence_info.sequence_id}: {e}")
                continue

            try:
                (
                    object_mesh_meshes,
                    _object_mesh_verts,
                    _object_mesh_faces,
                    object_surface_points,
                    object_surface_normals,
                    compute_tips_dist,
                ) = self.load_object_meshes(sequence_info, device)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {sequence_info.sequence_id}: {e}")
                continue

            if args.visualize and viser_server is not None:
                for handle in viser_object_handles.values():
                    handle.remove()
                viser_object_handles.clear()
                for part in sequence_info.object_body_names:
                    if part not in object_mesh_meshes:
                        continue
                    mesh = object_mesh_meshes[part]
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    viser_object_handles[part] = viser_server.scene.add_mesh_trimesh(
                        name=f"/object/{part}",
                        mesh=mesh,
                        position=np.array([0.0, 0.0, 0.0]),
                        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    )

            if args.save:
                object_mesh_radius = [
                    (
                        object_surface_points[body_name]
                        - object_surface_points[body_name].mean(dim=0)
                    )
                    .norm(dim=1)
                    .max()
                    .item()
                    for body_name in sequence_info.object_body_names
                ]
                logger_data = ManoSharpaData(
                    sequence_id=sequence_info.sequence_id,
                    raw_motion_file=sequence_info.raw_motion_file,
                    object_name=sequence_info.object_name,
                    safe_object_name=make_usd_safe(sequence_info.object_name),
                    robot_name="sharpa_wave",
                    fps=self.get_fps(),
                    mano_flat_hand_mean=mano_kwargs.get("flat_hand_mean", True),
                    mano_center_idx=mano_kwargs.get("center_idx", None),
                    mano_right_betas=raw_data["right_betas"].tolist(),
                    mano_left_betas=raw_data["left_betas"].tolist(),
                    mano_to_robot_scale=None,
                    right_robot_finger_joint_names=[],
                    right_robot_frame_names=[],
                    right_robot_frame_task_names=[],
                    left_robot_finger_joint_names=[],
                    left_robot_frame_names=[],
                    left_robot_frame_task_names=[],
                    object_body_names=sequence_info.object_body_names,
                    safe_object_body_names=[
                        make_usd_safe(name) for name in sequence_info.object_body_names
                    ],
                    object_mesh_paths=self.get_object_mesh_paths(sequence_info),
                    object_urdf_paths=self.get_object_urdf_paths(sequence_info),
                    object_mesh_radius=object_mesh_radius,
                    mano_link_names=list(MANO_HAND_LINKS.keys()),
                )

            mano_results: dict[str, Any] = {}
            for side in ("right", "left"):
                mano_results[side] = mano.forward(
                    side=side,
                    global_orient=raw_data[f"{side}_global_orient"],
                    finger_pose=raw_data[f"{side}_finger_pose"],
                    transl=raw_data[f"{side}_trans"],
                    betas=raw_data[f"{side}_betas"],
                )

            frame_start, frame_end = self.get_frame_range(H)
            for frame_id in range(frame_start, frame_end):
                joints: dict[str, torch.Tensor] = {}
                joints_wxyz: dict[str, torch.Tensor] = {}
                for side in ("right", "left"):
                    joints[side] = mano_results[side]["joints"][frame_id]
                    joints_wxyz[side] = mano_results[side]["joints_wxyz"][frame_id]

                frame_poses = self.get_frame_object_poses(
                    frame_id,
                    object_data,
                    object_surface_points,
                    object_surface_normals,
                    device,
                )

                if args.visualize and viser_server is not None:
                    for side in ("right", "left"):
                        mano.visualize(
                            viser_server,
                            side,
                            vertices=mano_results[side]["vertices"][frame_id],
                            faces=mano_results[side]["faces"],
                            joints=joints[side],
                            joints_wxyz=joints_wxyz[side],
                        )
                    for idx, part in enumerate(sequence_info.object_body_names):
                        if part in viser_object_handles and idx < len(
                            frame_poses.object_body_position
                        ):
                            pos = frame_poses.object_body_position[idx]
                            wxyz = frame_poses.object_body_wxyz[idx]
                            viser_object_handles[part].position = (
                                np.asarray(pos)
                                if not isinstance(pos, np.ndarray)
                                else pos
                            )
                            viser_object_handles[part].wxyz = (
                                np.asarray(wxyz)
                                if not isinstance(wxyz, np.ndarray)
                                else wxyz
                            )

                tips_dist: dict[str, Any] = {}
                if (
                    compute_tips_dist
                    and frame_poses.object_surface_points_world is not None
                ):
                    for side in ("right", "left"):
                        tips_dist[side] = compute_tip_to_object_surface_distance(
                            joints[side],
                            frame_poses.object_surface_points_world,
                        )

                # Compute contact positions on hand links, object surface, contact normals, and part ids.
                contact_data: dict[str, dict[str, torch.Tensor]] = {}
                if frame_poses.object_surface_points_world is not None:
                    for handle in viser_contact_handles:
                        handle.remove()
                    viser_contact_handles.clear()
                    for side in ("right", "left"):
                        hand_verts = mano_results[side]["vertices"][frame_id]
                        hand_faces = mano_results[side]["faces"]
                        hand_normals = trimesh.Trimesh(
                            vertices=hand_verts.cpu().numpy(),
                            faces=hand_faces.cpu().numpy(),
                        ).vertex_normals
                        hand_normals = (
                            torch.from_numpy(hand_normals).float().to(device)
                        )  # point outward
                        # TODO (xzhu): store all contact points and normals
                        (
                            _,
                            _,
                            object_contact_part_ids,
                            hand_contact_points_world,
                            hand_contact_normals_world,
                            contact_dists,
                        ) = approximate_contact_with_id(
                            frame_poses.object_surface_points_world,
                            frame_poses.object_surface_normals_world,
                            frame_poses.object_surface_points_part_ids,
                            hand_verts,
                            hand_normals,
                            threshold=0.01,
                        )
                        # Reduce the number of contacts to NUM_MANO_LINKS by averaging the contacts on the same link.
                        hand_link_contact_positions = torch.zeros(
                            (NUM_MANO_LINKS, 3), device=device
                        )
                        hand_link_contact_normals = torch.zeros(
                            (NUM_MANO_LINKS, 3), device=device
                        )
                        hand_link_contact_part_ids = torch.zeros(
                            (NUM_MANO_LINKS,), device=device
                        )
                        object_contact_positions = torch.zeros(
                            (NUM_MANO_LINKS, 3), device=device
                        )
                        object_contact_normals = torch.zeros(
                            (NUM_MANO_LINKS, 3), device=device
                        )
                        if len(contact_dists) > 0:
                            (
                                hand_link_contact_positions,
                                hand_link_contact_normals,
                                hand_link_contact_part_ids,
                            ) = compute_hand_link_contact_positions(
                                joints[side],
                                object_contact_part_ids,
                                hand_contact_points_world,
                                hand_contact_normals_world,
                                contact_dists,
                            )
                            object_contact_positions, object_contact_normals = (
                                find_object_contact_positions(
                                    hand_link_contact_positions,
                                    frame_poses.object_surface_points_world,
                                    frame_poses.object_surface_normals_world,
                                )
                            )

                            if args.visualize and viser_server is not None:
                                for link_idx in range(NUM_MANO_LINKS):
                                    if object_contact_positions[link_idx].norm() < 1e-3:
                                        continue
                                    hand_link_name = list(MANO_HAND_LINKS.keys())[
                                        link_idx
                                    ]
                                    object_contact_handle = viser_server.scene.add_icosphere(
                                        name=f"/object/contacts/{side}/{hand_link_name}",
                                        position=object_contact_positions[link_idx]
                                        .cpu()
                                        .numpy(),
                                        radius=0.003,
                                        color=np.array([0, 0, 255]),
                                    )
                                    viser_contact_handles.append(object_contact_handle)
                                    hand_link_contact_handle = viser_server.scene.add_icosphere(
                                        name=f"/mano/{side}/contacts/{hand_link_name}",
                                        position=hand_link_contact_positions[link_idx]
                                        .cpu()
                                        .numpy(),
                                        radius=0.003,
                                        color=np.array([0, 255, 0]),
                                    )
                                    viser_contact_handles.append(
                                        hand_link_contact_handle
                                    )
                                normal_lines = torch.cat(
                                    [
                                        hand_link_contact_positions.unsqueeze(1),
                                        (
                                            hand_link_contact_positions
                                            + hand_link_contact_normals * 0.01
                                        ).unsqueeze(1),
                                    ],
                                    dim=1,
                                )
                                hand_link_contact_normal_handle = (
                                    viser_server.scene.add_line_segments(
                                        name=f"/mano/{side}/contacts/normals",
                                        points=normal_lines.cpu().numpy(),
                                        colors=np.zeros_like(
                                            normal_lines.cpu().numpy()
                                        ),
                                        line_width=2.0,
                                    )
                                )
                                viser_contact_handles.append(
                                    hand_link_contact_normal_handle
                                )

                        contact_data[side] = {
                            "hand_link_contact_positions": hand_link_contact_positions,
                            "hand_link_contact_normals": hand_link_contact_normals,
                            "object_contact_positions": object_contact_positions,
                            "object_contact_normals": object_contact_normals,
                            "hand_link_contact_part_ids": hand_link_contact_part_ids,
                        }

                if args.save:
                    logger_data.log_timestep(
                        mano_right_trans=raw_data["right_trans"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_right_global_orient=raw_data["right_global_orient"][
                            frame_id
                        ]
                        .cpu()
                        .tolist(),
                        mano_right_finger_pose=raw_data["right_finger_pose"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_right_joints=joints["right"].cpu().tolist(),
                        mano_right_joints_wxyz=joints_wxyz["right"].cpu().tolist(),
                        mano_right_fitting_err=raw_data["right_fitting_err"][frame_id]
                        .cpu()
                        .item(),
                        mano_right_tips_distance=tips_dist.get("right"),
                        mano_right_link_contact_positions=contact_data["right"][
                            "hand_link_contact_positions"
                        ]
                        .cpu()
                        .tolist(),
                        mano_right_link_contact_normals=contact_data["right"][
                            "hand_link_contact_normals"
                        ]
                        .cpu()
                        .tolist(),
                        mano_right_object_contact_positions=contact_data["right"][
                            "object_contact_positions"
                        ]
                        .cpu()
                        .tolist(),
                        mano_right_object_contact_normals=contact_data["right"][
                            "object_contact_normals"
                        ]
                        .cpu()
                        .tolist(),
                        mano_right_object_contact_part_ids=contact_data["right"][
                            "hand_link_contact_part_ids"
                        ]
                        .cpu()
                        .tolist(),
                        mano_left_trans=raw_data["left_trans"][frame_id].cpu().tolist(),
                        mano_left_global_orient=raw_data["left_global_orient"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_left_finger_pose=raw_data["left_finger_pose"][frame_id]
                        .cpu()
                        .tolist(),
                        mano_left_joints=joints["left"].cpu().tolist(),
                        mano_left_joints_wxyz=joints_wxyz["left"].cpu().tolist(),
                        mano_left_fitting_err=raw_data["left_fitting_err"][frame_id]
                        .cpu()
                        .item(),
                        mano_left_tips_distance=tips_dist.get("left"),
                        mano_left_link_contact_positions=contact_data["left"][
                            "hand_link_contact_positions"
                        ]
                        .cpu()
                        .tolist(),
                        mano_left_link_contact_normals=contact_data["left"][
                            "hand_link_contact_normals"
                        ]
                        .cpu()
                        .tolist(),
                        mano_left_object_contact_positions=contact_data["left"][
                            "object_contact_positions"
                        ]
                        .cpu()
                        .tolist(),
                        mano_left_object_contact_normals=contact_data["left"][
                            "object_contact_normals"
                        ]
                        .cpu()
                        .tolist(),
                        mano_left_object_contact_part_ids=contact_data["left"][
                            "hand_link_contact_part_ids"
                        ]
                        .cpu()
                        .tolist(),
                        object_articulation=frame_poses.object_articulation,
                        object_root_axis_angle=frame_poses.object_root_axis_angle,
                        object_root_position=frame_poses.object_root_position,
                        object_body_position=frame_poses.object_body_position,
                        object_body_wxyz=frame_poses.object_body_wxyz,
                    )

            if args.save:
                logger_data.save_to_parquet(
                    root_path=str(args.output_dir),
                    partition_cols=DEFAULT_PARTITION_COLS,
                )
