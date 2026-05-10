"""Support-surface reconstruction from object still-poses.

Auto-detects schema (motion_v1 / mano_sharpa) from parquet columns, finds
frames where each object body is at rest, and writes a USD file of disk
prims that the sim spawner can load. Used both by the standalone CLI
(``scripts/reconstruct_support_surfaces.py``) and by the planner
(``planner/g1_planner.py``) to regenerate surfaces for transformed object
trajectories.
"""

from __future__ import annotations

import colorsys
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.motion_schema import load_motion_data_parquet
from robotic_grounding.retarget.data_logger import ManoSharpaData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISK_HEIGHT = 0.01  # thin disk thickness in meters
DISK_RADIUS_SCALE = (
    1.0  # multiplicative bloat on the computed radius; > 1.0 enlarges disks
)
GROUND_Z_THRESHOLD = 0.05  # disks below this Z are on the ground plane


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_local_mesh_path(path: str) -> str | None:
    """Resolve a possibly-stale absolute path to a local file under ``ASSET_DIR``.

    Parquets emitted from container builds carry absolute Docker mesh paths
    (``/workspace/.../assets/meshes/...``) that don't exist locally. We
    take the suffix after ``assets/meshes/`` and re-root it under the
    repo's local ``ASSET_DIR/meshes/``.
    """
    if not path:
        return None
    if Path(path).exists():
        return path
    if "assets/meshes/" not in path:
        return None
    suffix = path.rsplit("assets/meshes/", maxsplit=1)[-1]
    local = Path(ASSET_DIR) / "meshes" / suffix
    if local.exists():
        return str(local)
    tex = local.parent / "mesh_tex.obj"
    if tex.exists():
        return str(tex)
    return None


def _load_object_meshes_from_paths(
    object_mesh_paths: list[str],
    object_body_names: list[str],
) -> dict[str, trimesh.Trimesh]:
    """Load object meshes from schema paths (one per body).

    Paths ending with ``_cm.obj`` are scaled by 0.01 (cm -> m). Stale
    container paths are remapped to the local ``ASSET_DIR``.
    """
    meshes: dict[str, trimesh.Trimesh] = {}
    for part, path in zip(object_body_names, object_mesh_paths, strict=True):
        resolved = _resolve_local_mesh_path(path)
        if resolved is None:
            continue
        mesh = trimesh.load(resolved)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if resolved.endswith("_cm.obj"):
            mesh.vertices *= 0.01
        meshes[part] = mesh
    return meshes


def _detect_parquet_schema(input_dir: Path) -> str:
    """Return ``"motion_v1"`` or ``"mano_sharpa"`` for a parquet dataset."""
    first = next(Path(input_dir).rglob("*.parquet"), None)
    if first is None:
        raise FileNotFoundError(f"No parquet files under {input_dir}")
    columns = set(pq.ParquetFile(str(first)).schema.names)
    if "schema_version" in columns:
        return "motion_v1"
    return "mano_sharpa"


def _load_parquet_data(input_dir: Path, sequence_id: str, schema: str) -> Any:
    """Load one sequence from Parquet using the appropriate data logger class."""
    filters = [("sequence_id", "=", sequence_id)]
    if schema == "motion_v1":
        partition_dir = Path(input_dir) / f"sequence_id={sequence_id}"
        inner = next(partition_dir.glob("robot_name=*"), None)
        if inner is None:
            raise FileNotFoundError(f"No robot_name=* partition under {partition_dir}")
        return load_motion_data_parquet(str(inner))
    return ManoSharpaData.from_parquet(str(input_dir), filters=filters)


def load_object_mesh_and_poses(
    input_dir: Path,
    sequence_id: str,
    schema: str | None = None,
) -> tuple[Any, dict[str, trimesh.Trimesh]]:
    """Load one sequence and its object meshes.

    Returns:
        ``(data, object_meshes)`` where ``data`` exposes ``object_body_position``,
        ``object_body_wxyz``, etc., and ``object_meshes`` maps body name to
        a ``trimesh.Trimesh``.
    """
    if schema is None:
        schema = _detect_parquet_schema(input_dir)
    data = _load_parquet_data(input_dir, sequence_id, schema)
    object_mesh_paths = getattr(data, "object_mesh_paths", None) or []
    object_body_names = getattr(data, "object_body_names", None) or []
    if object_mesh_paths and len(object_mesh_paths) == len(object_body_names):
        object_meshes = _load_object_meshes_from_paths(
            object_mesh_paths,
            object_body_names,
        )
    else:
        object_meshes = {}
    return data, object_meshes


# ---------------------------------------------------------------------------
# Still-frame detection
# ---------------------------------------------------------------------------


def _frames_where_object_still(
    positions: np.ndarray,
    quats_wxyz: np.ndarray | None = None,
    pos_threshold_m: float = 0.001,
    angle_threshold_rad: float = 0.01,
    min_consecutive: int = 5,
) -> np.ndarray:
    """Return frame indices where the object is still.

    A frame is locally still when its inter-frame displacement and orientation
    change are both under the given thresholds. Only runs of at least
    ``min_consecutive`` locally-still frames are returned.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_frames = positions.shape[0]
    if n_frames == 0:
        return np.array([], dtype=np.intp)
    if n_frames == 1:
        return np.array([0], dtype=np.intp)

    pos_delta = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    locally_still = np.zeros(n_frames, dtype=bool)
    locally_still[0] = pos_delta[0] < pos_threshold_m
    locally_still[1:] = pos_delta < pos_threshold_m

    if quats_wxyz is not None:
        quats = np.asarray(quats_wxyz, dtype=np.float64)
        if quats.shape[0] != n_frames or quats.shape[1] != 4:
            raise ValueError("quats_wxyz must be (N, 4) with N = len(positions)")
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        dot = np.abs(np.sum(quats[1:] * quats[:-1], axis=1))
        dot = np.clip(dot, 0.0, 1.0)
        angle_delta = 2.0 * np.arccos(dot)
        still_by_angle = np.zeros(n_frames, dtype=bool)
        still_by_angle[0] = angle_delta[0] < angle_threshold_rad
        still_by_angle[1:] = angle_delta < angle_threshold_rad
        locally_still &= still_by_angle

    if min_consecutive <= 1:
        return np.nonzero(locally_still)[0]

    result = np.zeros(n_frames, dtype=bool)
    run_start = -1
    for i in range(n_frames):
        if locally_still[i]:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0 and (i - run_start) >= min_consecutive:
                result[run_start:i] = True
            run_start = -1
    if run_start >= 0 and (n_frames - run_start) >= min_consecutive:
        result[run_start:n_frames] = True

    return np.nonzero(result)[0]


def still_frames_from_object_data(
    data: Any,
    body_index: int = 0,
    pos_threshold_m: float = 0.001,
    angle_threshold_rad: float = 0.01,
) -> np.ndarray:
    """Return frame indices where the given object body is still."""
    obj_pos = data.object_body_position
    obj_quat = data.object_body_wxyz
    if hasattr(obj_pos, "cpu"):
        obj_pos = obj_pos.cpu().numpy()
    if hasattr(obj_quat, "cpu"):
        obj_quat = obj_quat.cpu().numpy()
    positions = np.array(obj_pos, dtype=np.float64)
    quats = np.array(obj_quat, dtype=np.float64)
    positions = positions[:, body_index, :]
    quats = quats[:, body_index, :]
    return _frames_where_object_still(
        positions,
        quats_wxyz=quats,
        pos_threshold_m=pos_threshold_m,
        angle_threshold_rad=angle_threshold_rad,
    )


# Alias kept for callers using the older name.
still_frames_from_mano_sharpa_data = still_frames_from_object_data


def extract_continuous_segments(still_frames: np.ndarray) -> list[np.ndarray]:
    """Split sorted still-frame indices into contiguous runs."""
    if len(still_frames) == 0:
        return []
    gaps = np.where(np.diff(still_frames) > 1)[0] + 1
    return np.split(still_frames, gaps)


# ---------------------------------------------------------------------------
# Support-disk computation
# ---------------------------------------------------------------------------


def _transform_vertices_to_world(
    vertices: np.ndarray,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Rotate and translate local mesh vertices into world frame."""
    w, x, y, z = quat_wxyz
    rot = Rotation.from_quat([x, y, z, w])
    return rot.apply(vertices) + np.asarray(position)


def compute_support_disk(
    vertices_world: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute a flat support disk enclosing the X-Y footprint of vertices.

    Returns ``(cx, cy, z, radius)`` — center of the X-Y AABB, minimum vertex
    Z, and half the larger of the X/Y spans (scaled by ``DISK_RADIUS_SCALE``).
    """
    xs = vertices_world[:, 0]
    ys = vertices_world[:, 1]
    zs = vertices_world[:, 2]
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    radius = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 * DISK_RADIUS_SCALE
    z = float(zs.min())
    return cx, cy, z, radius


def _enclosing_circle_of_two(
    cx1: float,
    cy1: float,
    r1: float,
    cx2: float,
    cy2: float,
    r2: float,
) -> tuple[float, float, float]:
    """Minimum enclosing circle of two circles."""
    dx = cx2 - cx1
    dy = cy2 - cy1
    d = np.hypot(dx, dy)
    if d + r2 <= r1:
        return cx1, cy1, r1
    if d + r1 <= r2:
        return cx2, cy2, r2
    r_new = (d + r1 + r2) / 2.0
    ratio = (r_new - r1) / d if d > 0 else 0.5
    cx_new = cx1 + dx * ratio
    cy_new = cy1 + dy * ratio
    return cx_new, cy_new, r_new


def merge_overlapping_disks(
    disks: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Iteratively merge disks that overlap in the X-Y plane.

    Each disk is ``(cx, cy, z, radius)``. The merged disk takes the minimum
    Z of the pair.
    """
    merged = list(disks)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                cx1, cy1, z1, r1 = merged[i]
                cx2, cy2, z2, r2 = merged[j]
                dist = np.hypot(cx2 - cx1, cy2 - cy1)
                if dist < r1 + r2:
                    cx_n, cy_n, r_n = _enclosing_circle_of_two(
                        cx1,
                        cy1,
                        r1,
                        cx2,
                        cy2,
                        r2,
                    )
                    merged[i] = (cx_n, cy_n, min(z1, z2), r_n)
                    merged.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return merged


# ---------------------------------------------------------------------------
# USD output
# ---------------------------------------------------------------------------


def _random_pastel_color() -> Gf.Vec3f:
    """Return a random pastel RGB color as a ``Gf.Vec3f``."""
    h = random.random()
    s = random.uniform(0.25, 0.45)
    v = random.uniform(0.85, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return Gf.Vec3f(r, g, b)


def create_disk(
    stage: Usd.Stage,
    prim_path: str,
    radius: float,
    center: tuple[float, float, float],
    height: float = DISK_HEIGHT,
) -> UsdGeom.Cylinder:
    """Add a single Z-axis cylinder disk to a USD stage."""
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateHeightAttr(height)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateDisplayColorAttr([_random_pastel_color()])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    cx, cy, cz = center
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set((cx, cy, cz - height / 2.0))
    return cyl


def write_support_surfaces_usd(
    all_disks: dict[str, list[tuple[float, float, float, float]]],
    output_path: str,
) -> Usd.Stage:
    """Write all support-surface disks into a single USD file."""
    output = Path(output_path)
    if output.exists():
        output.unlink()
    stage = Usd.Stage.CreateNew(str(output))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/support_surfaces")
    stage.SetDefaultPrim(root.GetPrim())

    disk_idx = 0
    for body_name, disks in all_disks.items():
        for cx, cy, z, radius in disks:
            safe_name = "".join(
                c if c.isalnum() or c == "_" else "_" for c in body_name
            )
            if safe_name and safe_name[0].isdigit():
                safe_name = f"_{safe_name}"
            prim_path = f"/support_surfaces/{safe_name}_{disk_idx}"
            create_disk(stage, prim_path, radius=radius, center=(cx, cy, z))
            disk_idx += 1

    stage.Save()
    return stage


# ---------------------------------------------------------------------------
# Sequence-level driver
# ---------------------------------------------------------------------------


def _compute_height_offset(data: Any, schema: str) -> float:
    """Z offset to align parquet data with the spawned scene.

    Object and support-surface positions from the retarget pipeline are
    already ground-relative (Z=0 at foot level), so no offset is needed.
    """
    return 0.0


def reconstruct_support_for_sequence(
    input_dir: Path,
    sequence_id: str,
    output_override: str | None = None,
    schema: str | None = None,
    ground_z_threshold: float = GROUND_Z_THRESHOLD,
) -> None:
    """Detect still frames, compute support disks, and write a USD file.

    Args:
        input_dir: Parquet root containing ``sequence_id=<...>`` partitions.
        sequence_id: Sequence identifier to process.
        output_override: Optional explicit path for the .usda output;
            otherwise written to
            ``<input_dir.parent>/reconstructed_stage/<sequence_id>_support.usda``.
        schema: Force a schema (``motion_v1`` or ``mano_sharpa``); auto-detected
            from parquet columns when ``None``.
        ground_z_threshold: Disks with z <= this are skipped (the sim ground
            plane already covers them).
    """
    if schema is None:
        schema = _detect_parquet_schema(input_dir)
    data, object_meshes = load_object_mesh_and_poses(
        input_dir, sequence_id, schema=schema
    )

    height_offset = _compute_height_offset(data, schema)
    if abs(height_offset) > 1e-6:
        print(f"  Applying height offset: {height_offset:.4f}m (schema={schema})")

    object_body_names = getattr(data, "object_body_names", None) or []
    num_frames = len(getattr(data, "object_body_position", []))
    print(f"\nSequence: {sequence_id}")
    print(f"Frames: {num_frames}")
    print(f"Object bodies: {object_body_names}")
    for name in object_body_names:
        mesh = object_meshes.get(name)
        n_verts = mesh.vertices.shape[0] if mesh is not None else 0
        print(f"  {name}: mesh vertices={n_verts}")

    if num_frames == 0 or not object_body_names:
        print("No object trajectory data — skipping.")
        return

    # For articulated objects, only reconstruct support for the root body
    # (index 0). Child bodies are connected via joints and don't rest on
    # surfaces.
    articulation = getattr(data, "object_articulation", None)
    is_articulated = articulation is not None and np.any(np.array(articulation) != 0.0)

    all_disks: dict[str, list[tuple[float, float, float, float]]] = {}

    for body_idx, body_name in enumerate(object_body_names):
        if is_articulated and body_idx > 0:
            print(f"  Skipping {body_name}: child body of articulated object")
            continue
        mesh = object_meshes.get(body_name)
        if mesh is None:
            print(f"  Skipping {body_name}: no mesh loaded")
            continue

        still_frames = still_frames_from_object_data(data, body_index=body_idx)
        segments = extract_continuous_segments(still_frames)
        print(
            f"  {body_name}: {len(still_frames)} still frames, "
            f"{len(segments)} segment(s)"
        )

        body_disks: list[tuple[float, float, float, float]] = []
        for seg in segments:
            first_frame = int(seg[0])
            pos = np.asarray(
                data.object_body_position[first_frame][body_idx],
                dtype=np.float64,
            )
            quat = np.asarray(
                data.object_body_wxyz[first_frame][body_idx],
                dtype=np.float64,
            )
            verts_world = _transform_vertices_to_world(mesh.vertices, pos, quat)
            disk = compute_support_disk(verts_world)
            body_disks.append(disk)

        if body_disks:
            merged = merge_overlapping_disks(body_disks)
            if abs(height_offset) > 1e-6:
                merged = [(cx, cy, z + height_offset, r) for cx, cy, z, r in merged]
            above_ground = [d for d in merged if d[2] > ground_z_threshold]
            on_ground = len(merged) - len(above_ground)
            if on_ground:
                print(
                    f"    filtered {on_ground} disk(s) at ground level "
                    f"(z <= {ground_z_threshold:.3f}m)"
                )
            print(
                f"    merged {len(body_disks)} disk(s) -> {len(merged)}, "
                f"kept {len(above_ground)} above ground"
            )
            if above_ground:
                all_disks[body_name] = above_ground

    if not all_disks:
        print("No support disks needed (object on ground or always held).")
        return

    total = sum(len(v) for v in all_disks.values())
    default_output_dir = input_dir.parent / "reconstructed_stage"
    default_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_override or str(
        default_output_dir / f"{sequence_id}_support.usda"
    )
    write_support_surfaces_usd(all_disks, output_path)
    print(f"Wrote {total} support surface(s) to {output_path}")
