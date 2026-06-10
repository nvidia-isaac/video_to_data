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


def _looks_like_sequence_subfolder(name: str) -> bool:
    """Return true for dataset sequence folders that contain per-sequence dirs."""
    return (
        name in {"arctic_processed", "taco_processed", "hot3d_processed_filtered"}
        or "_processed" in name
    )


def _legacy_motion_layout(motion_file: str | Path) -> dict | None:
    """Parse legacy .../<processed>/<sequence>/<robot> motion layouts."""
    path = Path(motion_file).resolve()
    motion_dir = path if path.is_dir() else path.parent
    if not motion_dir.name or motion_dir.name.startswith("robot_name="):
        return None
    seq_dir = motion_dir.parent
    sequence_subfolder = seq_dir.parent
    if seq_dir.name.startswith("sequence_id="):
        return None
    if not _looks_like_sequence_subfolder(sequence_subfolder.name):
        return None
    return {
        "robot_name": unquote(motion_dir.name),
        "sequence_id": unquote(seq_dir.name),
        "motion_folder": str(motion_dir),
        "sequence_subfolder": sequence_subfolder,
    }


# Per-object scale overrides for OakInk-V2 objects whose meshes are repaired as
# solid blobs and need a small uniform scale tweak to avoid PhysX interpenetration.
# Ported verbatim from ManipTrans (main/dataset/oakink2_dataset_utils.py:34-41).
# Keys use OakInk-V2's @ notation; lookup also accepts the body_name "safe" form
# (underscores) used downstream. Effect: shrink the body, grow the cap, opening
# ~10mm of radial clearance — enough for the cap to wrap the body's neck.
OAKINK2_OBJECT_SCALE: dict[str, float] = {
    "O02@0206@00001": 1.15,  # alcohol burner — cap (grow 15%)
    "O02@0206@00002": 0.95,  # alcohol burner — body (shrink 5%)
    "O02@0029@00011": 0.9,
    "O02@0029@00012": 1.2,
    "O02@0015@00020": 0.98,
    "O02@0015@00019": 1.02,
}


# Per-object collision-approximation overrides. Defaults to URDF importer's
# convex-hull/decomposition; "sdf" forces a PhysX SDF mesh collider via the
# `apply_sdf_collision_approximations` prestartup event. Use this for objects
# where the visual mesh has real concavity (open cavities, hollow vessels, etc.)
# that convex decomposition can't represent — fingertips reaching into the cavity
# correctly experience no contact instead of hitting phantom hull material.
OAKINK2_OBJECT_COLLISION_APPROXIMATION: dict[str, str] = {
    "O02@0011@00003": "sdf",  # pour_tube vessel — open-top, non-watertight mesh
    "O02@0206@00001": "sdf",  # uncap_alcohol_burner cap — 23-piece convex too slow at 4096 envs
}


def _oakink2_collision_approximation_for(body_name: str) -> str | None:
    """Look up collision-approximation override for an OakInk-V2 object."""
    if body_name in OAKINK2_OBJECT_COLLISION_APPROXIMATION:
        return OAKINK2_OBJECT_COLLISION_APPROXIMATION[body_name]
    at_form = body_name.replace("_", "@", 2) if body_name.startswith("O02_") else None
    if at_form and at_form in OAKINK2_OBJECT_COLLISION_APPROXIMATION:
        return OAKINK2_OBJECT_COLLISION_APPROXIMATION[at_form]
    return None


def _oakink2_scale_for(body_name: str) -> tuple[float, float, float]:
    """Look up uniform scale for an OakInk-V2 object by body name.

    Accepts either the @-notation key (``O02@0206@00002``) or the underscore-safe
    body_name form (``O02_0206_00002``) — both map to the same scale entry.
    Returns ``(1.0, 1.0, 1.0)`` when no override is registered.
    """
    if body_name in OAKINK2_OBJECT_SCALE:
        s = OAKINK2_OBJECT_SCALE[body_name]
    else:
        at_form = (
            body_name.replace("_", "@", 2) if body_name.startswith("O02_") else None
        )
        if at_form and at_form in OAKINK2_OBJECT_SCALE:
            s = OAKINK2_OBJECT_SCALE[at_form]
        else:
            return (1.0, 1.0, 1.0)
    return (s, s, s)


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

    # Per-object collision approximation override. None → use default (convex hull /
    # decomposition from URDF importer). "sdf" → apply PhysX SDF mesh collider in
    # the prestartup event (preserves cavity geometry for hollow / non-watertight objects).
    collision_approximation: str | None = None


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
        """Extract robot_name, sequence_id, motion_folder, motion_filters from motion path."""
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

        legacy = _legacy_motion_layout(path)
        if legacy:
            result["robot_name"] = legacy["robot_name"]
            result["sequence_id"] = legacy["sequence_id"]
            result["motion_folder"] = legacy["motion_folder"]
            result["motion_filters"] = []
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
        if obj_name:
            spec = get_object_spec(obj_name)
            if spec and spec.urdf_path:
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
            # e.g. meshes/hot3d/12345.glb -> urdfs/hot3d/12345_rigid.urdf.
            # Only overwrite when the derived path actually exists; otherwise
            # keep the original parquet urdf_path so the dataset_root fallback
            # below has a filename to search for on OSMO.
            if not urdf_path or not Path(urdf_path).exists():
                mesh_derived = cls._urdf_from_mesh_path(
                    mesh_paths[i] if i < len(mesh_paths) else None
                )
                if mesh_derived:
                    urdf_path = mesh_derived

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

            obj = ObjectConfig(
                name=body_name,
                usd_path=urdf_path,
                scale=_oakink2_scale_for(body_name),
                collision_approximation=_oakink2_collision_approximation_for(body_name),
            )
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
        """Derive dataset root from a partitioned or legacy motion file path.

        Partitioned example:
          .../<dataset>/taco_processed/sequence_id=.../robot_name=... -> .../<dataset>/
        Legacy example:
          .../<dataset>/taco/taco_processed/<sequence>/sharpa_wave -> .../<dataset>/
        """
        path = Path(motion_file)

        legacy = _legacy_motion_layout(path)
        if legacy:
            sequence_subfolder = legacy["sequence_subfolder"]
            candidates = [sequence_subfolder.parent, sequence_subfolder.parent.parent]
            marker_dirs = ("taco_urdfs", "reconstructed_stage", "hot3d_urdfs", "urdfs")
            for candidate in candidates:
                if (
                    candidate
                    and candidate.exists()
                    and any((candidate / m).exists() for m in marker_dirs)
                ):
                    return candidate
            return (
                candidates[0]
                if candidates and candidates[0] != candidates[0].parent
                else None
            )

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
        """Search for an asset file in bounded dataset subdirectories."""
        if not dataset_root or not dataset_root.is_dir():
            return None
        frontier = [dataset_root]
        for _depth in range(3):
            next_frontier: list[Path] = []
            for root in frontier:
                candidate = root / filename
                if candidate.exists():
                    return str(candidate)
                try:
                    next_frontier.extend(p for p in root.iterdir() if p.is_dir())
                except OSError:
                    continue
            frontier = next_frontier
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
        """Auto-discover fixed objects (support surfaces) from the motion file path.

        For maniptrans_oakink sequences (microwave/laptop articulated), the
        per-sequence support-disk reconstruction is buggy — disks land inside
        the object mesh causing spawn-time depenetration kicks. Those sequences
        use a single ManipTrans-style fixed table instead, spawned by
        apply_scene_config — so we skip the disk discovery entirely here.
        """
        fixed: list[ObjectConfig] = []
        # Only the ARTICULATED maniptrans_oakink subset uses the fixed table
        # path (see _add_maniptrans_oakink_table). Rigid maniptrans_oakink
        # sequences (in maniptrans_oakink_processed/, not _articulated/) keep
        # the standard per-sequence support-disk discovery below.
        if "maniptrans_oakink_processed_articulated" in motion_file:
            return fixed  # fixed table spawned by apply_scene_config._add_maniptrans_oakink_table
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
    """Find reconstructed support surface USDA from partitioned parquet path.

    Primary lookup: walk up to ``sequence_id=<name>`` and check
    ``../../reconstructed_stage/<name>_support.usda`` relative to it.

    Fallback for OSMO: on OSMO the motion_file lives under
    ``/osmo/data/input/0/<dataset>/`` so the relative walk fails.
    We maintain a mapping from known dataset subfolder names to the
    repo-resident ``reconstructed_stage/`` directories (baked into the
    Docker image) so support surfaces load correctly during OSMO training.
    """
    # Known dataset subfolder → repo-resident reconstructed_stage/ dir.
    # These paths are relative to HUMAN_MOTION_DATA_DIR (ASSET_DIR/human_motion_data).
    _DATASET_STAGE_DIRS: dict[str, Path] = {
        "spider_oakink_processed": Path(HUMAN_MOTION_DATA_DIR)
        / "spider_oakink"
        / "reconstructed_stage",
        "spider_oakink_processed_invalid": Path(HUMAN_MOTION_DATA_DIR)
        / "spider_oakink"
        / "reconstructed_stage",
        "spider_oakinkv2_new_processed": Path(HUMAN_MOTION_DATA_DIR)
        / "spider_oakinkv2_new"
        / "reconstructed_stage",
        "maniptrans_oakink_processed": Path(HUMAN_MOTION_DATA_DIR)
        / "maniptrans_oakink"
        / "reconstructed_stage",
        "maniptrans_oakink_processed_articulated": Path(HUMAN_MOTION_DATA_DIR)
        / "maniptrans_oakink"
        / "reconstructed_stage",
    }

    path = Path(motion_file).resolve()
    for parent in [path] + list(path.parents):
        if parent.name.startswith("sequence_id="):
            seq_id = parent.name.split("=", 1)[1]
            # Primary: check sibling reconstructed_stage/ (local runs)
            stage_dir = parent.parent.parent / "reconstructed_stage"
            support_path = stage_dir / f"{seq_id}_support.usda"
            if support_path.exists():
                return str(support_path)
            # Fallback: check repo-resident stage dir by dataset subfolder name
            dataset_subfolder = parent.parent.name  # e.g. "spider_oakink_processed"
            if dataset_subfolder in _DATASET_STAGE_DIRS:
                fallback = (
                    _DATASET_STAGE_DIRS[dataset_subfolder] / f"{seq_id}_support.usda"
                )
                if fallback.exists():
                    return str(fallback)
            return None

    legacy = _legacy_motion_layout(path)
    if legacy:
        seq_id = str(legacy["sequence_id"])
        sequence_subfolder = legacy["sequence_subfolder"]
        candidates = [
            sequence_subfolder.parent
            / "reconstructed_stage"
            / f"{seq_id}_support.usda",
            sequence_subfolder.parent.parent
            / "reconstructed_stage"
            / f"{seq_id}_support.usda",
        ]
        dataset_subfolder = sequence_subfolder.name
        if dataset_subfolder in _DATASET_STAGE_DIRS:
            candidates.append(
                _DATASET_STAGE_DIRS[dataset_subfolder] / f"{seq_id}_support.usda"
            )
        for support_path in candidates:
            if support_path.exists():
                return str(support_path)
    return None
