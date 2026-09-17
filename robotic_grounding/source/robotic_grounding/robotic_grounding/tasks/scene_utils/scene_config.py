# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.assets.object_registry import is_articulated

HUMAN_MOTION_DATA_DIR = os.environ.get(
    "HUMAN_MOTION_DATA_DIR", os.path.join(ASSET_DIR, "human_motion_data")
)
URDF_DIR = os.path.join(ASSET_DIR, "urdfs")


@dataclass
class ObjectConfig:
    """Configuration for a rigid scene object (target or fixed)."""

    name: str
    usd_path: str
    position_key: str = ""
    quaternion_key: str = ""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pos_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    init_pos: list[float] | None = None
    init_rot: list[float] | None = None


@dataclass
class ArticulatedObjectConfig:
    """Configuration for an articulated (multi-body) scene object loaded from URDF."""

    name: str
    urdf_path: str
    body_names: list[str] = field(default_factory=list)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pos_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    init_pos: list[float] | None = None
    init_rot: list[float] | None = None  # wxyz quaternion

    body_init_positions: list[list[float]] | None = None
    body_init_rotations: list[list[float]] | None = None  # wxyz quaternions

    init_joint_pos: float | None = None


@dataclass
class SceneConfig:
    """Scene configuration auto-discovered from parquet data.

    ``scene_objects`` contains tracked objects and may be empty for robot-only
    reference tracking. When present, ``scene_objects[0]`` is the primary
    object used for command tracking and contact sensors. Fixed scene objects
    are independent and are always spawned.
    """

    motion_file: str
    episode_length_s: float
    scene_objects: list[ObjectConfig | ArticulatedObjectConfig]
    fixed_objects: list[ObjectConfig]

    # Auto-discovered from partition path
    robot_name: str | None = None
    sequence_id: str | None = None
    motion_folder: str | None = None
    motion_filters: list[tuple[str, str, str]] | None = None
    object_body_names: list[str] | None = None

    @classmethod
    def from_motion_file(cls, motion_file: str) -> SceneConfig:
        """Build a SceneConfig from a parquet motion file path. Everything is auto-discovered."""
        motion_file = cls._resolve_motion_file(motion_file)
        data = pq.read_table(motion_file).to_pydict()
        partition = cls._parse_partition_path(motion_file)

        # Fail fast: check required assets exist before Isaac Sim loads objects
        cls._validate_assets(data, motion_file)

        object_body_names = (
            data.get("safe_object_body_names", [[]])[0]
            or data.get("object_body_names", [[]])[0]
            or None
        )
        scene_objects: list[ObjectConfig | ArticulatedObjectConfig] = []
        if object_body_names:
            object_type = cls._detect_object_type(data)
            scene_objects = cls._build_scene_objects(data, object_type, motion_file)
        fixed_objects = cls._build_fixed_objects(motion_file)
        episode_length_s = cls._build_episode_length_s(data)

        return cls(
            motion_file=motion_file,
            episode_length_s=episode_length_s,
            scene_objects=scene_objects,
            fixed_objects=fixed_objects,
            robot_name=partition.get("robot_name"),
            sequence_id=partition.get("sequence_id"),
            motion_folder=partition.get("motion_folder"),
            motion_filters=partition.get("motion_filters"),
            object_body_names=object_body_names,
        )

    @staticmethod
    def _resolve_motion_file(raw_path: str) -> str:
        """Resolve a motion file path.

        Accepts:
        - Full path to a parquet file or partitioned dir
        - dataset/dataset_retargeted/sequence_id/robot_name like "arctic/arctic_processed/dataset_s01_ketchup_use_01/sharpa_wave"
        """
        motion_file = raw_path
        if not Path(motion_file).is_absolute():
            motion_file = str(Path.cwd() / motion_file)

        # A relative path may be given against the data root — including the full
        # ".../sequence_id=.../robot_name=..." partition form.
        if not Path(motion_file).exists() and not Path(raw_path).is_absolute():
            under_root = os.path.join(HUMAN_MOTION_DATA_DIR, raw_path.strip("/"))
            if Path(under_root).exists():
                motion_file = under_root

        # Bare dataset/dataset_retargeted/sequence_id/robot_name shorthand.
        if not Path(motion_file).exists():
            parts = raw_path.strip("/").split("/")
            if len(parts) == 4 and not any("=" in p for p in parts):
                dataset, dataset_retargeted, seq_id, robot = parts
                motion_file = os.path.join(
                    HUMAN_MOTION_DATA_DIR,
                    dataset,
                    dataset_retargeted,
                    f"sequence_id={seq_id}",
                    f"robot_name={robot}",
                )

        if not Path(motion_file).exists():
            raise FileNotFoundError(
                f"Motion file not found: {raw_path} (resolved: {motion_file})"
            )

        return motion_file

    @staticmethod
    def _parse_partition_path(motion_file: str) -> dict:
        """Extract robot_name, sequence_id, motion_folder, motion_filters from partition path."""
        result: dict = {}
        path = Path(motion_file).resolve()
        for parent in [path] + list(path.parents):
            name = parent.name
            if name.startswith("robot_name="):
                result["robot_name"] = unquote(name.split("=", 1)[1])
            elif name.startswith("sequence_id="):
                result["sequence_id"] = unquote(name.split("=", 1)[1])
                result["motion_folder"] = str(parent.parent)

        if "robot_name" in result and "sequence_id" in result:
            result["motion_filters"] = [
                ("robot_name", "=", result["robot_name"]),
                ("sequence_id", "=", result["sequence_id"]),
            ]
        return result

    @staticmethod
    def _detect_object_type(data: dict) -> str:
        """Detect whether the scene object is articulated or rigid.

        Checks the object registry first — if the object has a urdf_path it is
        articulated, regardless of whether articulation values are zero
        (e.g. grab sequences where the lid never moves).
        Exception: if body_names == ["object"], this is the rigid URDF link
        convention used by Arctic "rigid_*" sequences, so treat as rigid even
        when the registry has an art URDF.
        Multiple rigid bodies (TACO tool+target, OakInk2 multi-object) have no
        urdf_path in the registry and fall through to "rigid".
        """
        obj_name = (
            data.get("safe_object_name", [None])[0]
            or data.get("object_name", [None])[0]
        )
        if is_articulated(obj_name):
            body_names = (
                data.get("safe_object_body_names", [[]])[0]
                or data.get("object_body_names", [[]])[0]
                or []
            )
            if body_names != ["object"]:
                return "articulated"
        return "rigid"

    @classmethod
    def _build_scene_objects(
        cls, data: dict, object_type: str, motion_file: str = ""
    ) -> list[ObjectConfig | ArticulatedObjectConfig]:
        """Build all scene objects from parquet data.

        For articulated objects (Arctic), builds a single ArticulatedObjectConfig.
        For rigid objects (TACO/OakInk2), builds one ObjectConfig per body.
        """
        if object_type == "articulated":
            return [cls._build_articulated_object(data, motion_file)]

        body_names = (
            data.get("safe_object_body_names", [[]])[0]
            or data.get("object_body_names", [[]])[0]
            or []
        )
        if not body_names:
            return []
        urdf_paths = data.get("object_urdf_paths", [[]])[0] or []
        mesh_paths = data.get("object_mesh_paths", [[]])[0] or []
        obj_name = (
            data.get("safe_object_name", [None])[0]
            or data.get("object_name", [None])[0]
        )
        dataset_root = (
            cls._dataset_root_from_motion_file(motion_file) if motion_file else None
        )
        objects: list[ObjectConfig | ArticulatedObjectConfig] = []

        for i, body_name in enumerate(body_names):

            # Resolve from parquet urdf_paths
            urdf_path = urdf_paths[i] if i < len(urdf_paths) else None

            # Re-root a baked /data/object_assets path under the motion file's dataset root
            if urdf_path and not Path(urdf_path).exists():
                urdf_path = (
                    cls._reroot_object_asset(urdf_path, dataset_root) or urdf_path
                )

            # Fallback: derive URDF path from mesh path by convention
            # e.g. meshes/hot3d/12345.glb -> urdfs/hot3d/12345_rigid.urdf
            if not urdf_path or not Path(urdf_path).exists():
                derived_urdf_path = cls._urdf_from_mesh_path(
                    mesh_paths[i] if i < len(mesh_paths) else None
                )
                if derived_urdf_path:
                    urdf_path = derived_urdf_path

            # Fallback: search for URDF by filename in the motion file's dataset
            if (
                (not urdf_path or not Path(urdf_path).exists())
                and dataset_root
                and urdf_path
            ):
                dataset_urdf = cls._find_asset_in_dataset(
                    Path(urdf_path).name, dataset_root
                )
                if dataset_urdf:
                    urdf_path = dataset_urdf

            assert urdf_path and Path(urdf_path).exists(), (
                f"Could not resolve rigid object for object_name='{obj_name}', "
                f"body='{body_name}'. Generate URDFs with scripts/generate_rigid_urdfs.py"
            )

            obj = ObjectConfig(name=body_name, usd_path=urdf_path)
            _load_body_pose(data, obj, i)
            objects.append(obj)

        return objects

    @staticmethod
    def _urdf_from_mesh_path(mesh_path: str | None) -> str | None:
        """Derive a rigid URDF path from an object mesh path by convention.

        Example: .../meshes/hot3d/12345.glb -> .../urdfs/hot3d/12345_rigid.urdf
        """
        if not mesh_path:
            return None
        mesh = Path(mesh_path)
        # Convention: urdfs/<dataset>/<stem>_rigid.urdf
        # Mesh is at meshes/<dataset>/<file>, URDF is at urdfs/<dataset>/<stem>_rigid.urdf
        dataset = mesh.parent.name
        urdf_path = Path(URDF_DIR) / dataset / f"{mesh.stem}_rigid.urdf"
        return str(urdf_path) if urdf_path.exists() else None

    @staticmethod
    def _dataset_root_from_motion_file(motion_file: str) -> Path | None:
        """Derive dataset root from a partitioned motion file path.

        Walks up the path past partition dirs (key=value format) and the
        sequences subfolder to reach the dataset root.

        Example:
          .../v2d_taco_retarget_exp_200/taco_processed/sequence_id=.../robot_name=...
          → .../v2d_taco_retarget_exp_200/
        """
        path = Path(motion_file)
        prev_no_eq = False
        for _ in range(10):
            if path == path.parent:
                return None
            has_eq = "=" in path.name
            if not has_eq and prev_no_eq:
                return path
            prev_no_eq = not has_eq
            path = path.parent
        return None

    @staticmethod
    def _find_asset_in_dataset(filename: str, dataset_root: Path) -> str | None:
        """Search for an asset file in immediate subdirectories of the dataset root."""
        if not dataset_root or not dataset_root.is_dir():
            return None
        direct = dataset_root / filename
        if direct.exists():
            return str(direct)
        for subdir in dataset_root.iterdir():
            if not subdir.is_dir():
                continue
            candidate = subdir / filename
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _reroot_object_asset(path: str | None, dataset_root: Path | None) -> str | None:
        """Re-root a path with an ``object_assets`` segment under the dataset root.

        Matches ``object_assets`` as a path *component*, not a substring, so a
        baked absolute path (``/data/object_assets/...``), a dataset-relative
        path (``arctic/object_assets/...``) and a bare-relative path
        (``object_assets/...``) all re-root the same way — and it stays correct
        if the stored paths later become relative. Returns the re-rooted path
        only if it exists.
        """
        if not path or dataset_root is None:
            return None
        parts = Path(path).parts
        if "object_assets" not in parts:
            return None
        idx = parts.index("object_assets")
        candidate = Path(dataset_root).joinpath(*parts[idx:])
        return str(candidate) if candidate.exists() else None

    @staticmethod
    def _validate_assets(data: dict, motion_file: str) -> None:
        """Check that required asset files exist before building the scene.

        Raises FileNotFoundError with an actionable message if any URDF or
        mesh file explicitly referenced by the parquet is missing. This
        catches errors early — before Isaac Sim spends time loading —
        rather than crashing mid-startup.

        Note: this only validates paths stored in the parquet. Objects
        resolved via the object registry or the mesh-derived URDF fallback
        are validated later in ``_build_scene_objects``.

        For URDFs, falls back to searching in the motion file's dataset root
        (e.g. OSMO-mounted dataset taco_urdfs/ subfolder) when the workspace
        path is absent.
        """
        urdf_paths = data.get("object_urdf_paths", [[]])[0] or []
        mesh_paths = data.get("object_mesh_paths", [[]])[0] or []
        missing: list[str] = []

        dataset_root = SceneConfig._dataset_root_from_motion_file(motion_file)

        for p in urdf_paths:
            if p and not Path(p).exists():
                resolved = SceneConfig._reroot_object_asset(p, dataset_root) or (
                    SceneConfig._find_asset_in_dataset(Path(p).name, dataset_root)
                    if dataset_root
                    else None
                )
                if not resolved:
                    missing.append(f"URDF: {p}")
        for p in mesh_paths:
            if p and not Path(p).exists():
                resolved = SceneConfig._reroot_object_asset(p, dataset_root) or (
                    SceneConfig._find_asset_in_dataset(Path(p).name, dataset_root)
                    if dataset_root
                    else None
                )
                if not resolved:
                    missing.append(f"Mesh: {p}")

        if missing:
            raise FileNotFoundError(
                f"Missing assets for motion file {motion_file}:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\n\nFix: python scripts/generate_rigid_urdfs.py --dataset <name>"
            )

    @classmethod
    def _build_articulated_object(
        cls, data: dict, motion_file: str = ""
    ) -> ArticulatedObjectConfig:
        """Build an articulated object, resolving its URDF from the dataset assets."""
        obj_name = (
            data.get("safe_object_name", [None])[0]
            or data.get("object_name", [None])[0]
        )
        if not obj_name:
            raise ValueError("Could not discover object_name from parquet")

        mesh_paths = data.get("object_mesh_paths", [[]])[0] or []
        dataset_root = (
            cls._dataset_root_from_motion_file(motion_file) if motion_file else None
        )
        urdf_path = (
            cls._articulated_urdf_from_mesh_path(mesh_paths[0]) if mesh_paths else None
        )
        if urdf_path and not Path(urdf_path).exists():
            urdf_path = cls._reroot_object_asset(urdf_path, dataset_root) or urdf_path
        if not urdf_path or not Path(urdf_path).exists():
            raise FileNotFoundError(
                f"Articulated URDF for '{obj_name}' not found (derived: {urdf_path}). "
                "Expected object_assets/urdfs/<dataset>/<object>.urdf."
            )

        obj = ArticulatedObjectConfig(name=obj_name, urdf_path=urdf_path)
        _load_articulated_poses(data, obj)
        return obj

    @staticmethod
    def _articulated_urdf_from_mesh_path(mesh_path: str | None) -> str | None:
        """Map a baked object mesh path to its sibling articulated URDF path.

        ``.../object_assets/meshes/<ds>/<obj>/x.obj`` maps to
        ``.../object_assets/urdfs/<ds>/<obj>.urdf``.

        Anchors on the ``object_assets`` path component rather than a fixed
        parent depth, so it derives the same URDF whether the mesh path is
        absolute or relative. Returns None if the path doesn't carry the
        expected ``object_assets/meshes/<ds>/<obj>/`` layout (a shorter/relative
        path no longer raises IndexError).
        """
        if not mesh_path:
            return None
        parts = Path(mesh_path).parts
        if "object_assets" not in parts:
            return None
        idx = parts.index("object_assets")
        tail = parts[idx + 1 :]  # (meshes, <ds>, <obj>, <file>)
        if len(tail) < 4 or tail[0] != "meshes":
            return None
        base = Path(*parts[: idx + 1])  # up to and including object_assets
        return str(base / "urdfs" / tail[1] / f"{tail[2]}.urdf")

    @staticmethod
    def _build_fixed_objects(motion_file: str) -> list[ObjectConfig]:
        """Auto-discover fixed objects (support surfaces) from the motion file path."""
        fixed: list[ObjectConfig] = []
        support_path = _discover_support_surface(motion_file)
        if support_path is not None:
            fixed.append(
                ObjectConfig(
                    name="support_surface",
                    usd_path=support_path,
                    init_pos=[0.0, 0.0, 0.0],
                    init_rot=[1.0, 0.0, 0.0, 0.0],
                )
            )
        return fixed

    @staticmethod
    def _build_episode_length_s(data: dict) -> float:
        """Build episode length from the first available reference trajectory."""
        try:
            fps = float(data.get("fps", [30.0])[0])
            if fps <= 0.0:
                return 20.0
            for field_name in (
                "robot_joint_positions",
                "robot_root_position",
                "ee_pose_w",
                "object_body_position",
            ):
                values = data.get(field_name, [None])[0]
                if values is not None and len(values) > 0:
                    return float(len(values) / fps)
        except (IndexError, TypeError, ValueError):
            pass
        return 20.0


# Parquet pose loading


def _load_articulated_poses(data: dict, obj: ArticulatedObjectConfig) -> None:
    """Populate an ArticulatedObjectConfig with frame-0 poses from parquet."""
    offset = obj.pos_offset

    if not obj.body_names:
        names = (
            data.get("safe_object_body_names", [[]])[0]
            or data.get("object_body_names", [[]])[0]
            or []
        )
        if names:
            obj.body_names = list(names)

    root_pos = data.get("object_root_position")
    if root_pos and root_pos[0]:
        pos = list(root_pos[0][0])
        obj.init_pos = [p + o for p, o in zip(pos, offset, strict=True)]

    root_aa = data.get("object_root_axis_angle")
    if root_aa and root_aa[0]:
        aa = np.array(root_aa[0][0])
        obj.init_rot = R.from_rotvec(aa).as_quat(scalar_first=True).tolist()

    body_pos = data.get("object_body_position")
    if body_pos and body_pos[0]:
        frame0 = body_pos[0][0]
        obj.body_init_positions = [
            [p + o for p, o in zip(bp, offset, strict=True)] for bp in frame0
        ]
        body0_pos = list(frame0[0])
        obj.init_pos = [p + o for p, o in zip(body0_pos, offset, strict=True)]

    body_rot = data.get("object_body_wxyz")
    if body_rot and body_rot[0]:
        frame0 = body_rot[0][0]
        obj.body_init_rotations = [list(bw) for bw in frame0]
        obj.init_rot = list(frame0[0])

    art = data.get("object_articulation")
    if art and art[0]:
        obj.init_joint_pos = float(art[0][0])


def _load_body_pose(data: dict, obj: ObjectConfig, body_index: int) -> None:
    """Load frame-0 pose for a specific body index from parquet body arrays."""
    offset = obj.pos_offset

    body_pos = data.get("object_body_position")
    if body_pos and body_pos[0]:
        frame0 = body_pos[0][0]
        if body_index < len(frame0):
            pos = list(frame0[body_index])
            obj.init_pos = [p + o for p, o in zip(pos, offset, strict=True)]

    body_rot = data.get("object_body_wxyz")
    if body_rot and body_rot[0]:
        frame0 = body_rot[0][0]
        if body_index < len(frame0):
            obj.init_rot = list(frame0[body_index])


def _discover_support_surface(motion_file: str) -> str | None:
    """Find reconstructed support surface USDA from partitioned parquet path.

    A sequence may be retargeted to several embodiments whose placement worlds differ
    in scale (the whole-body retarget shrinks the scene into the robot's workspace,
    the floating-hand one does not), so a single shared surface cannot serve both.
    Prefer a robot-specific ``<seq>_<robot>_support.usda`` when present and fall back
    to the shared ``<seq>_support.usda``.
    """
    path = Path(motion_file).resolve()
    parents = [path] + list(path.parents)
    robot_name = next(
        (p.name.split("=", 1)[1] for p in parents if p.name.startswith("robot_name=")),
        None,
    )
    for parent in parents:
        if parent.name.startswith("sequence_id="):
            seq_id = parent.name.split("=", 1)[1]
            stage_dir = parent.parent.parent / "reconstructed_stage"
            candidates = [stage_dir / f"{seq_id}_support.usda"]
            if robot_name:
                candidates.insert(0, stage_dir / f"{seq_id}_{robot_name}_support.usda")
            for support_path in candidates:
                if support_path.exists():
                    return str(support_path)
            return None
    return None


def discover_motion_files(motion_dir: str, robot_name: str | None = None) -> list[str]:
    """Expand a partitioned motion folder into one motion path per sequence.

    Given a ``<dataset>_processed`` directory laid out as
    ``sequence_id=<id>/robot_name=<robot>`` partitions, return the partition
    path for every sequence of the selected robot, sorted by sequence id. This
    is the ``--motion_dir`` counterpart to ``--motion_file``: it feeds a whole
    bank of reference motions to a multi-motion command (see
    ``MotionTrackingCommand``).

    Args:
        motion_dir: Directory containing ``sequence_id=*/robot_name=*``
            partitions. Accepts an absolute path or a path relative to the
            human-motion-data asset root (same shorthand as ``--motion_file``).
        robot_name: Restrict to this robot. If ``None`` and the directory holds
            exactly one robot, that robot is used; multiple robots raise.

    Returns:
        Sorted list of per-sequence partition paths.

    Raises:
        FileNotFoundError: The directory or its partitions cannot be found.
        ValueError: The directory holds multiple robots and none was requested.
    """
    resolved = motion_dir
    if not Path(resolved).exists() and not Path(motion_dir).is_absolute():
        under_root = os.path.join(HUMAN_MOTION_DATA_DIR, motion_dir.strip("/"))
        if Path(under_root).exists():
            resolved = under_root
    if not Path(resolved).is_dir():
        raise FileNotFoundError(
            f"Motion directory not found: {motion_dir} (resolved: {resolved})"
        )

    partitions = sorted(Path(resolved).glob("sequence_id=*/robot_name=*"))
    if not partitions:
        raise FileNotFoundError(
            "No 'sequence_id=*/robot_name=*' partitions found under "
            f"{resolved}. Point --motion_dir at a <dataset>_processed folder."
        )

    robots = sorted({p.name.split("=", 1)[1] for p in partitions})
    if robot_name is None:
        if len(robots) > 1:
            raise ValueError(
                f"Motion directory {resolved} contains multiple robots "
                f"{robots}; narrow --motion_dir to a single-robot folder."
            )
        robot_name = robots[0]

    selected = sorted(
        str(p) for p in partitions if p.name == f"robot_name={robot_name}"
    )
    if not selected:
        raise FileNotFoundError(
            f"No partitions for robot_name={robot_name} under {resolved} "
            f"(available robots: {robots})."
        )
    return selected
