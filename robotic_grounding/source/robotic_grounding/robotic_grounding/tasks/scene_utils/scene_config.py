from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.assets.object_registry import get_object_spec

HUMAN_MOTION_DATA_DIR = os.path.join(ASSET_DIR, "human_motion_data")
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

    ``scene_objects[0]`` is the primary object used for command tracking and
    contact sensors. All objects are spawned into the scene.
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

        object_type = cls._detect_object_type(data)
        scene_objects = cls._build_scene_objects(data, object_type, motion_file)
        fixed_objects = cls._build_fixed_objects(motion_file)
        object_body_names = (
            data.get("safe_object_body_names", [[]])[0]
            or data.get("object_body_names", [[]])[0]
            or None
        )
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
        - dataset/dataset_retargeted/sequence_id/robot_name like "arctic/arctic_processed/arctic_s01_ketchup_use_01/sharpa_wave"
        """
        motion_file = raw_path
        if not Path(motion_file).is_absolute():
            motion_file = str(Path.cwd() / motion_file)

        if not Path(motion_file).exists():
            parts = raw_path.strip("/").split("/")
            if len(parts) == 4:
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
        always articulated, regardless of whether articulation values are zero
        (e.g. grab sequences where the lid never moves).
        Multiple rigid bodies (TACO tool+target, OakInk2 multi-object) have no
        urdf_path in the registry and fall through to "rigid".
        """
        obj_name = (
            data.get("safe_object_name", [None])[0]
            or data.get("object_name", [None])[0]
        )
        if obj_name:
            spec = get_object_spec(obj_name)
            if spec and spec.urdf_path:
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
            return [cls._build_articulated_object(data)]

        body_names = (
            data.get("safe_object_body_names", [[]])[0]
            or data.get("object_body_names", [[]])[0]
            or []
        )
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

            # Try registry first (for first body with known object name)
            if i == 0 and obj_name:
                spec = get_object_spec(obj_name)
                if spec and (spec.rigid_urdf_path or spec.usd_path):
                    asset_path = spec.rigid_urdf_path or spec.usd_path
                    obj = ObjectConfig(
                        name=body_name,
                        usd_path=asset_path,
                        scale=spec.scale,
                    )
                    _load_body_pose(data, obj, i)
                    objects.append(obj)
                    continue

            # Resolve from parquet urdf_paths
            urdf_path = urdf_paths[i] if i < len(urdf_paths) else None

            # Fallback: derive URDF path from mesh path by convention
            # e.g. meshes/hot3d/12345.glb -> urdfs/hot3d/12345_rigid.urdf
            if not urdf_path or not Path(urdf_path).exists():
                urdf_path = cls._urdf_from_mesh_path(
                    mesh_paths[i] if i < len(mesh_paths) else None
                )

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

        if not objects:
            raise ValueError("No scene objects could be built from parquet data")

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
                resolved = (
                    SceneConfig._find_asset_in_dataset(Path(p).name, dataset_root)
                    if dataset_root
                    else None
                )
                if not resolved:
                    missing.append(f"URDF: {p}")
        for p in mesh_paths:
            if p and not Path(p).exists():
                missing.append(f"Mesh: {p}")

        if missing:
            raise FileNotFoundError(
                f"Missing assets for motion file {motion_file}:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\n\nFix: python scripts/generate_rigid_urdfs.py --dataset <name>"
            )

    @staticmethod
    def _build_articulated_object(data: dict) -> ArticulatedObjectConfig:
        """Build an articulated object from parquet data and the object registry."""
        obj_name = (
            data.get("safe_object_name", [None])[0]
            or data.get("object_name", [None])[0]
        )
        if not obj_name:
            raise ValueError("Could not discover object_name from parquet")

        spec = get_object_spec(obj_name)
        if not spec or not spec.urdf_path:
            raise ValueError(f"No urdf_path for '{obj_name}' — add to object registry")

        obj = ArticulatedObjectConfig(name=obj_name, urdf_path=spec.urdf_path)
        _load_articulated_poses(data, obj)
        return obj

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
        """Build the episode length from the parquet data."""
        try:
            timesteps = len(data.get("object_articulation", [[]])[0])
            fps = data.get("fps", [30.0])[0]
            episode_length_s = float(timesteps / fps)
        except Exception:
            episode_length_s = 20.0
        return episode_length_s


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

    body_rot = data.get("object_body_wxyz")
    if body_rot and body_rot[0]:
        frame0 = body_rot[0][0]
        obj.body_init_rotations = [list(bw) for bw in frame0]

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
    """Find reconstructed support surface USDA from partitioned parquet path."""
    path = Path(motion_file).resolve()
    for parent in [path] + list(path.parents):
        if parent.name.startswith("sequence_id="):
            seq_id = parent.name.split("=", 1)[1]
            stage_dir = parent.parent.parent / "reconstructed_stage"
            support_path = stage_dir / f"{seq_id}_support.usda"
            if support_path.exists():
                return str(support_path)
            return None
    return None
