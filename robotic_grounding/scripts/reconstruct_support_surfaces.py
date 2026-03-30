#!/usr/bin/env python3
"""Reconstruct support surfaces from object still-poses in TACO/ARCTIC sequences.

Reads Parquet from the loader output; object meshes are loaded from the object_mesh_paths
field in the schema (one path per body), like vis_retargeted.py.

Usage:
  1. Run loader first: python scripts/retarget/taco_loader.py --save  (or arctic_loader.py --save)
  2. python scripts/reconstruct_support_surfaces.py --input_dir ... [--sequence_id ID]
"""

import argparse
import colorsys
import random
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData, list_sequence_ids
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR_TACO = HUMAN_MOTION_DATA_DIR / "taco_loaded"
DEFAULT_INPUT_DIR_ARCTIC = HUMAN_MOTION_DATA_DIR / "arctic_loaded"
DEFAULT_INPUT_DIR_OAKINK2 = HUMAN_MOTION_DATA_DIR / "oakink2_loaded"

DISK_HEIGHT = 0.01  # thin disk thickness in meters


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_object_meshes_from_paths(
    object_mesh_paths: list[str],
    object_body_names: list[str],
) -> dict[str, trimesh.Trimesh]:
    """Load object meshes from schema paths (one per body).

    Paths ending with _cm.obj are scaled by 0.01 (cm -> m).
    """
    meshes: dict[str, trimesh.Trimesh] = {}
    for part, path in zip(object_body_names, object_mesh_paths, strict=True):
        if not path or not Path(path).exists():
            continue
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if path.endswith("_cm.obj"):
            mesh.vertices *= 0.01
        meshes[part] = mesh
    return meshes


def load_object_mesh_and_poses(
    input_dir: Path,
    sequence_id: str,
) -> tuple[ManoSharpaData, dict[str, trimesh.Trimesh]]:
    """Load one sequence from Parquet and its object meshes via object_mesh_paths.

    Returns:
        data: ManoSharpaData with object_body_position, object_body_wxyz, etc.
        object_meshes: dict[body_name] -> trimesh.
    """
    data = ManoSharpaData.from_parquet(
        str(input_dir),
        filters=[("sequence_id", "=", sequence_id)],
    )
    object_mesh_paths = getattr(data, "object_mesh_paths", None) or []
    if object_mesh_paths and len(object_mesh_paths) == len(data.object_body_names):
        object_meshes = _load_object_meshes_from_paths(
            object_mesh_paths,
            data.object_body_names,
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
    """Return frame indices where the object is still, based on inter-frame motion.

    A frame is considered *locally* still when the displacement and orientation change
    relative to its predecessor are below the given thresholds.  To reduce false
    positives, a frame is only included in the output if it belongs to a run of at
    least ``min_consecutive`` locally-still frames.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_frames = positions.shape[0]
    if n_frames == 0:
        return np.array([], dtype=np.intp)
    if n_frames == 1:
        return np.array([0], dtype=np.intp)

    # Per-frame position stillness (frame-to-frame delta below threshold)
    pos_delta = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    locally_still = np.zeros(n_frames, dtype=bool)
    locally_still[0] = pos_delta[0] < pos_threshold_m
    locally_still[1:] = pos_delta < pos_threshold_m

    # Per-frame orientation stillness
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

    # Require runs of at least min_consecutive still frames
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
    # Handle run that extends to the last frame
    if run_start >= 0 and (n_frames - run_start) >= min_consecutive:
        result[run_start:n_frames] = True

    return np.nonzero(result)[0]


def still_frames_from_mano_sharpa_data(
    data: ManoSharpaData,
    body_index: int = 0,
    pos_threshold_m: float = 0.001,
    angle_threshold_rad: float = 0.01,
) -> np.ndarray:
    """Return frame indices where the given object body is still."""
    positions = np.array(data.object_body_position, dtype=np.float64)
    quats = np.array(data.object_body_wxyz, dtype=np.float64)
    positions = positions[:, body_index, :]
    quats = quats[:, body_index, :]
    return _frames_where_object_still(
        positions,
        quats_wxyz=quats,
        pos_threshold_m=pos_threshold_m,
        angle_threshold_rad=angle_threshold_rad,
    )


def extract_continuous_segments(still_frames: np.ndarray) -> list[np.ndarray]:
    """Split sorted still-frame indices into contiguous runs.

    E.g. ``[5, 6, 7, 20, 21]`` -> ``[array([5, 6, 7]), array([20, 21])]``.
    """
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
    rot = Rotation.from_quat([x, y, z, w])  # scipy expects xyzw
    return rot.apply(vertices) + np.asarray(position)


def compute_support_disk(
    vertices_world: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute a flat support disk enclosing the X-Y footprint of world-space vertices.

    Returns:
        ``(cx, cy, z, radius)`` -- center of the X-Y bounding box, minimum vertex Z,
        and half the larger of the X/Y spans.
    """
    xs = vertices_world[:, 0]
    ys = vertices_world[:, 1]
    zs = vertices_world[:, 2]
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    radius = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
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
    """Minimum enclosing circle of two circles. Returns ``(cx, cy, r)``."""
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

    Each disk is ``(cx, cy, z, radius)``.  Two disks overlap when the distance
    between their X-Y centers is less than the sum of their radii.  The merged
    disk takes the minimum *z* of the pair.
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
    """Return a random pastel RGB color as a ``Gf.Vec3f`` with components in [0, 1]."""
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
    """Add a single disk (thin Z-axis cylinder) to an existing USD stage.

    The disk is assigned a randomized pastel display color.

    Args:
        stage: The USD stage to add the disk to.
        prim_path: USD prim path for the cylinder (e.g. ``"/support_surfaces/bowl_0"``).
        radius: Radius of the disk in the X-Y plane.
        center: ``(x, y, z)`` position.  The disk is translated so its top face sits at *z*
            (i.e. the cylinder center is offset down by ``height / 2``).
        height: Thickness of the disk along Z.

    Returns:
        The created UsdGeom.Cylinder.
    """
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
    """Write all support-surface disks into a single USD file.

    Args:
        all_disks: Mapping from body_name to list of ``(cx, cy, z, radius)`` tuples.
        output_path: File path for the .usda/.usdc output.

    Returns:
        The saved Usd.Stage.
    """
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
            prim_path = f"/support_surfaces/{safe_name}_{disk_idx}"
            create_disk(stage, prim_path, radius=radius, center=(cx, cy, z))
            disk_idx += 1

    stage.Save()
    return stage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct support surfaces from object still-poses.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Parquet root (e.g. taco_loaded/mano_object_only or arctic_loaded/mano_object_only).",
    )
    parser.add_argument(
        "--dataset",
        choices=("taco", "arctic", "oakink2"),
        default="oakink2",
        help="Dataset for default input_dir when --input_dir not set (default: taco).",
    )
    parser.add_argument(
        "--sequence_id",
        type=str,
        default=None,
        help="Sequence to load; if omitted, run on all sequences.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list sequence IDs and exit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .usda path for support surfaces (default: <sequence_id>_support.usda).",
    )
    return parser.parse_args()


def _process_sequence(
    input_dir: Path, sequence_id: str, output_override: str | None
) -> None:
    """Process a single sequence: detect still frames, compute disks, write USD."""
    data, object_meshes = load_object_mesh_and_poses(input_dir, sequence_id)

    num_frames = len(data.mano_right_trans)
    print(f"\nSequence: {sequence_id}")
    print(f"Frames: {num_frames}")
    print(f"Object bodies: {data.object_body_names}")
    for name in data.object_body_names:
        mesh = object_meshes.get(name)
        n_verts = mesh.vertices.shape[0] if mesh is not None else 0
        print(f"  {name}: mesh vertices={n_verts}")

    # For articulated objects, only reconstruct support surfaces for the root body
    # (index 0). Child bodies are connected via joints and don't rest on surfaces.
    articulation = getattr(data, "object_articulation", None)
    is_articulated = articulation is not None and np.any(np.array(articulation) != 0.0)

    all_disks: dict[str, list[tuple[float, float, float, float]]] = {}

    for body_idx, body_name in enumerate(data.object_body_names):
        if is_articulated and body_idx > 0:
            print(f"  Skipping {body_name}: child body of articulated object")
            continue
        mesh = object_meshes.get(body_name)
        if mesh is None:
            print(f"  Skipping {body_name}: no mesh loaded")
            continue

        still_frames = still_frames_from_mano_sharpa_data(data, body_index=body_idx)
        segments = extract_continuous_segments(still_frames)
        print(
            f"  {body_name}: {len(still_frames)} still frames, {len(segments)} segment(s)"
        )

        body_disks: list[tuple[float, float, float, float]] = []
        for seg in segments:
            first_frame = int(seg[0])
            pos = np.asarray(
                data.object_body_position[first_frame][body_idx], dtype=np.float64
            )
            quat = np.asarray(
                data.object_body_wxyz[first_frame][body_idx], dtype=np.float64
            )
            verts_world = _transform_vertices_to_world(mesh.vertices, pos, quat)
            disk = compute_support_disk(verts_world)
            body_disks.append(disk)
            print(
                f"    segment frames {seg[0]}-{seg[-1]}: "
                f"disk center=({disk[0]:.4f}, {disk[1]:.4f}), z={disk[2]:.4f}, r={disk[3]:.4f}"
            )

        if body_disks:
            merged = merge_overlapping_disks(body_disks)
            print(f"    merged {len(body_disks)} disk(s) -> {len(merged)}")
            all_disks[body_name] = merged

    if not all_disks:
        print("No support disks generated.")
        return

    total = sum(len(v) for v in all_disks.values())
    default_output_dir = input_dir.parent / "reconstructed_stage"
    default_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_override or str(
        default_output_dir / f"{sequence_id}_support.usda"
    )
    write_support_surfaces_usd(all_disks, output_path)
    print(f"Wrote {total} support surface(s) to {output_path}")


def main() -> None:
    """Entry point: parse CLI args and reconstruct support surfaces for the specified dataset."""
    args = _parse_args()
    if args.input_dir:
        input_dir = args.input_dir
    elif args.dataset == "taco":
        input_dir = DEFAULT_INPUT_DIR_TACO
    elif args.dataset == "arctic":
        input_dir = DEFAULT_INPUT_DIR_ARCTIC
    else:
        input_dir = DEFAULT_INPUT_DIR_OAKINK2
    if not input_dir.is_dir():
        print(
            f"Input dir not found: {input_dir}. Run the loader first (e.g. taco_loader.py --save)."
        )
        return

    sequence_ids = list_sequence_ids(str(input_dir))
    if not sequence_ids:
        print(f"No sequences in {input_dir}")
        return

    if args.list:
        for sid in sequence_ids:
            print(sid)
        return

    if args.sequence_id:
        if args.sequence_id not in sequence_ids:
            print(f"Sequence {args.sequence_id} not found. Available: {sequence_ids}")
            return
        ids_to_process = [args.sequence_id]
    else:
        ids_to_process = sequence_ids
        print(
            f"No --sequence_id given, processing all {len(ids_to_process)} sequence(s)."
        )

    for sequence_id in ids_to_process:
        _process_sequence(input_dir, sequence_id, args.output)


if __name__ == "__main__":
    main()
