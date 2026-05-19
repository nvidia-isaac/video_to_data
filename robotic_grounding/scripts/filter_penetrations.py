#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

r"""Filter HOT3D processed sequences with excessive hand penetration.

Reads from hot3d_processed/ (partitioned Parquet) and copies sequences
with max penetration ≤ threshold to an output directory.

Two penetration types are checked per frame:
  hand-object  — robot finger capsule/sphere into the object mesh
  hand-hand    — right hand capsule/sphere into left hand capsule/sphere

Penetration depth is measured as the depth of surface overlap in metres.
Sequences where the per-frame maximum exceeds ``--max_penetration`` (default
0.02 m = 2 cm) in ANY frame are rejected.

Usage
-----
  python scripts/filter_penetrations.py \\
      --input_dir ~/datasets/.../hot3d_processed \\
      --output_dir ~/datasets/.../hot3d_processed_valid

  # Dry-run: report stats without copying
  python scripts/filter_penetrations.py \\
      --input_dir ~/datasets/.../hot3d_processed \\
      --output_dir ~/datasets/.../hot3d_processed_valid \\
      --dry_run

  # Faster (sample every 5th frame, 8 workers)
  python scripts/filter_penetrations.py \\
      --input_dir ~/datasets/.../hot3d_processed \\
      --output_dir ~/datasets/.../hot3d_processed_valid \\
      --stride 5 --num_workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URDF paths for sharpa_wave primitive collision geometry
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../video_to_data
_URDF_DIR = (
    _REPO_ROOT
    / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/sharpawave"
)
_URDF_RIGHT = _URDF_DIR / "right_sharpa_wave_primitive.urdf"
_URDF_LEFT = _URDF_DIR / "left_sharpa_wave_primitive.urdf"

# Mesh paths in the parquet use the Docker container prefix; remap to local.
_MESH_PATH_REMAPS = {
    "/workspace/video_to_data": str(_REPO_ROOT),
}


# ---------------------------------------------------------------------------
# Collision geometry data structures
# ---------------------------------------------------------------------------


@dataclass
class CollisionShape:
    """A primitive collision shape (sphere or capsule) attached to a robot link."""

    link_name: str
    radius: float
    # For capsules: the shaft endpoints in the link's local frame.
    # For spheres: both endpoints equal the sphere centre.
    ep1_local: np.ndarray  # (3,)
    ep2_local: np.ndarray  # (3,)


def _rpy_to_matrix(rpy: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Roll-pitch-yaw (XYZ extrinsic) → 3×3 rotation matrix."""
    return Rotation.from_euler("xyz", rpy).as_matrix()


def _parse_collision_shapes(urdf_path: Path) -> dict[str, CollisionShape]:
    """Return {link_name: CollisionShape} from a primitive URDF."""
    tree = ET.parse(str(urdf_path))
    shapes: dict[str, CollisionShape] = {}
    for link in tree.getroot().findall("link"):
        name = link.get("name", "")
        col = link.find("collision")
        if col is None:
            continue
        geom = col.find("geometry")
        origin = col.find("origin")
        if geom is None:
            continue

        xyz = (
            np.array(list(map(float, origin.get("xyz", "0 0 0").split())))
            if origin is not None
            else np.zeros(3)
        )
        rpy = (
            np.array(list(map(float, origin.get("rpy", "0 0 0").split())))
            if origin is not None
            else np.zeros(3)
        )
        R_col = _rpy_to_matrix(rpy)

        sphere = geom.find("sphere")
        capsule = geom.find("capsule")

        if sphere is not None:
            r = float(sphere.get("radius") or 0.0)
            shapes[name] = CollisionShape(name, r, xyz.copy(), xyz.copy())

        elif capsule is not None:
            r = float(capsule.get("radius") or 0.0)
            half = float(capsule.get("length") or 0.0) / 2.0
            # Capsule axis is along local Z in the collision frame; rotate by R_col,
            # then add the origin offset.
            axis_col = R_col @ np.array([0.0, 0.0, half])
            shapes[name] = CollisionShape(
                name,
                r,
                xyz + axis_col,
                xyz - axis_col,
            )

    return shapes


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _quat_wxyz_to_matrix(qwxyz: list | np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] → 3×3 rotation matrix."""
    qw, qx, qy, qz = qwxyz
    # scipy from_quat expects [x, y, z, w]
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _segment_segment_distance(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> float:
    """Minimum distance between two line segments (p1-p2) and (q1-q2)."""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    # Degenerate cases
    if a < 1e-10 and e < 1e-10:
        return float(np.linalg.norm(r))
    if a < 1e-10:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e < 1e-10:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    cp = p1 + s * d1 - (q1 + t * d2)
    return float(np.linalg.norm(cp))


# ---------------------------------------------------------------------------
# Per-sequence penetration computation
# ---------------------------------------------------------------------------


def _remap_mesh_path(path: str) -> str:
    for prefix, replacement in _MESH_PATH_REMAPS.items():
        if path.startswith(prefix):
            return replacement + path[len(prefix) :]
    return path


def _load_hull(mesh_path: str, cache: dict) -> tuple[trimesh.Trimesh | None, float]:
    """Load mesh and return (convex_hull, hull_volume_ratio).

    hull_volume_ratio = hull.volume / mesh.volume.  A ratio >> 1 means the
    object is highly concave/open (e.g. a cup, vase, or open glasses frame) and
    the convex hull is a poor proxy for the actual solid.  Callers should skip
    the hand-object check when this ratio exceeds a threshold.
    """
    key = os.path.basename(mesh_path)
    if key in cache:
        return cache[key]
    local_path = _remap_mesh_path(mesh_path)
    if not os.path.exists(local_path):
        log.warning("Mesh not found: %s", local_path)
        cache[key] = (None, 0.0)
        return cache[key]
    try:
        mesh = trimesh.load(local_path, force="mesh")
        hull = mesh.convex_hull
        ratio = (
            hull.volume / mesh.volume if mesh.volume and mesh.volume > 1e-10 else 999.0
        )
        cache[key] = (hull, ratio)
        return cache[key]
    except Exception as e:
        log.warning("Failed to load mesh %s: %s", local_path, e)
        cache[key] = (None, 0.0)
        return cache[key]


class _HandShapeCache:
    """Pre-computes per-link local-frame capsule endpoints for one hand."""

    def __init__(
        self, shapes: dict[str, CollisionShape], frame_names: list[str]
    ) -> None:
        # Ordered list of (frame_index, CollisionShape) for links that appear
        # in both the URDF and the parquet frame list.
        self.entries: list[tuple[int, CollisionShape]] = []
        name_to_idx = {n: i for i, n in enumerate(frame_names)}
        for link_name, shape in shapes.items():
            idx = name_to_idx.get(link_name)
            if idx is not None:
                self.entries.append((idx, shape))

    def world_spheres(
        self, frames_t: list
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Return [(ep1_world, ep2_world, radius), ...] for frame t's link poses."""
        result: list[tuple[np.ndarray, np.ndarray, float]] = []
        for idx, shape in self.entries:
            pose = frames_t[idx]  # [px, py, pz, qw, qx, qy, qz]
            pos = np.array(pose[:3], dtype=float)
            R = _quat_wxyz_to_matrix(pose[3:7])
            # Transform both capsule endpoints to world space
            ep1_w = pos + R @ shape.ep1_local
            ep2_w = pos + R @ shape.ep2_local
            result.append((ep1_w, ep2_w, shape.radius))
        return result


def _max_hand_object_penetration(
    capsules: list[tuple[np.ndarray, np.ndarray, float]],
    obj_hull: trimesh.Trimesh,
    obj_pos: np.ndarray,
    obj_R: np.ndarray,
) -> float:
    """Max penetration depth (m) of any capsule/sphere into the object hull.

    Points are transformed into the object's local frame so we can query a
    single static hull instead of re-transforming the mesh every frame.

    trimesh signed_distance convention: positive = inside, negative = outside.
    Penetration depth of a sphere of radius r at distance sd from surface:
        depth = max(0, r + sd)
    For a capsule, the representative points are both endpoints.
    """
    # Collect query points and matching radii
    pts = []
    radii = []
    for ep1_w, ep2_w, r in capsules:
        pts.append(ep1_w)
        pts.append(ep2_w)
        radii.extend([r, r])

    pts_arr = np.array(pts)
    radii_arr = np.array(radii)

    # Transform to object-local frame (avoids re-transforming hull every frame)
    pts_local = (obj_R.T @ (pts_arr - obj_pos).T).T

    sd = trimesh.proximity.signed_distance(obj_hull, pts_local)
    # depth = max(0, r + sd)  [sd positive = inside]
    depths = np.maximum(0.0, radii_arr + sd)
    return float(depths.max()) if len(depths) > 0 else 0.0


def _max_hand_hand_penetration(
    right_caps: list[tuple[np.ndarray, np.ndarray, float]],
    left_caps: list[tuple[np.ndarray, np.ndarray, float]],
) -> float:
    """Max capsule-capsule penetration depth between right and left hand."""
    max_pen = 0.0
    for ep1_r, ep2_r, r_r in right_caps:
        for ep1_l, ep2_l, r_l in left_caps:
            dist = _segment_segment_distance(ep1_r, ep2_r, ep1_l, ep2_l)
            pen = max(0.0, r_r + r_l - dist)
            max_pen = max(max_pen, pen)
    return max_pen


# ---------------------------------------------------------------------------
# Per-sequence entry point (used by multiprocessing)
# ---------------------------------------------------------------------------


def _check_sequence(
    args_tuple: tuple,
) -> dict:
    """Evaluate one sequence; return result dict.

    Args:
        args_tuple: (seq_dir, right_shapes, left_shapes, max_pen, hull_ratio_max, stride, hull_cache)
    """
    seq_dir, right_shapes, left_shapes, max_pen, hull_ratio_max, stride, hull_cache = (
        args_tuple
    )

    seq_id = seq_dir.parent.name.replace("sequence_id=", "")
    parquet_files = list(seq_dir.glob("*.parquet"))
    if not parquet_files:
        return {
            "seq_id": seq_id,
            "rejected": True,
            "reason": "no_parquet",
            "max_ho_pen": 0.0,
            "max_hh_pen": 0.0,
            "object_name": "",
            "seq_dir": seq_dir,
        }

    try:
        data = pq.read_table(str(parquet_files[0])).to_pydict()
    except Exception as e:
        return {
            "seq_id": seq_id,
            "rejected": True,
            "reason": f"read_error:{e}",
            "max_ho_pen": 0.0,
            "max_hh_pen": 0.0,
            "object_name": "",
            "seq_dir": seq_dir,
        }

    right_frames_seq = data.get("robot_right_frames", [[]])[0]
    left_frames_seq = data.get("robot_left_frames", [[]])[0]
    right_frame_names = data.get("right_robot_frame_names", [[]])[0]
    left_frame_names = data.get("left_robot_frame_names", [[]])[0]
    obj_positions = data.get("object_body_position", [[]])[0]
    obj_wxyz = data.get("object_body_wxyz", [[]])[0]
    obj_mesh_paths = data.get("object_mesh_paths", [[]])[0]
    object_name = data.get("object_name", [""])[0] or ""

    n_frames = len(right_frames_seq)
    if n_frames == 0:
        return {
            "seq_id": seq_id,
            "rejected": False,
            "reason": "ok",
            "max_ho_pen": 0.0,
            "max_hh_pen": 0.0,
            "object_name": object_name,
            "seq_dir": seq_dir,
        }

    right_cache = _HandShapeCache(right_shapes, right_frame_names)
    left_cache = _HandShapeCache(left_shapes, left_frame_names)

    # Load object hulls (one per body; most sequences have 1 body).
    # Pairs of (hull_or_None, hull_ratio).  Skip bodies where hull_ratio
    # exceeds hull_ratio_max: highly concave objects (AR glasses, open vases)
    # produce massive false positives with the convex hull signed_distance check.
    hull_entries: list[tuple[trimesh.Trimesh | None, float]] = []
    for mp in obj_mesh_paths:
        hull, ratio = _load_hull(mp, hull_cache)
        if ratio > hull_ratio_max:
            hull_entries.append((None, ratio))  # skip this body
        else:
            hull_entries.append((hull, ratio))

    max_ho_pen = 0.0
    max_hh_pen = 0.0

    for t in range(0, n_frames, max(1, stride)):
        right_caps = right_cache.world_spheres(right_frames_seq[t])
        left_caps = left_cache.world_spheres(left_frames_seq[t])

        # Hand-object
        if obj_positions and len(obj_positions) > t:
            for body_idx, (hull, _) in enumerate(hull_entries):
                if hull is None:
                    continue
                if body_idx >= len(obj_positions[t]):
                    continue
                obj_pos = np.array(obj_positions[t][body_idx], dtype=float)
                obj_qwxyz = obj_wxyz[t][body_idx]
                obj_R = _quat_wxyz_to_matrix(obj_qwxyz)

                ho = _max_hand_object_penetration(
                    right_caps + left_caps, hull, obj_pos, obj_R
                )
                max_ho_pen = max(max_ho_pen, ho)

        # Hand-hand
        hh = _max_hand_hand_penetration(right_caps, left_caps)
        max_hh_pen = max(max_hh_pen, hh)

        # Early-exit once both thresholds exceeded
        if max_ho_pen > max_pen and max_hh_pen > max_pen:
            break

    overall_max = max(max_ho_pen, max_hh_pen)
    rejected = overall_max > max_pen
    reason = "ok"
    if rejected:
        parts = []
        if max_ho_pen > max_pen:
            parts.append(f"hand_object:{max_ho_pen*100:.1f}cm")
        if max_hh_pen > max_pen:
            parts.append(f"hand_hand:{max_hh_pen*100:.1f}cm")
        reason = ",".join(parts)

    return {
        "seq_id": seq_id,
        "rejected": rejected,
        "reason": reason,
        "max_ho_pen": max_ho_pen,
        "max_hh_pen": max_hh_pen,
        "object_name": object_name,
        "seq_dir": seq_dir,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _find_parquet_dirs(input_dir: Path) -> list[Path]:
    """Return all robot_name= subdirs under sequence_id= dirs."""
    dirs = []
    for seq_dir in sorted(input_dir.glob("sequence_id=*")):
        if not seq_dir.is_dir():
            continue
        for robot_dir in sorted(seq_dir.glob("robot_name=*")):
            if robot_dir.is_dir():
                dirs.append(robot_dir)
    return dirs


def main() -> None:
    """CLI entry point — parse args and run the penetration filter."""
    parser = argparse.ArgumentParser(
        description="Filter hot3d_processed sequences by penetration depth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Processed parquet directory (hot3d_processed/).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for valid sequences.",
    )
    parser.add_argument(
        "--max_penetration",
        type=float,
        default=0.02,
        help="Maximum allowed penetration in metres (default 0.02 = 2 cm).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Check every N-th frame (default 3; use 1 for every frame).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Parallel worker processes (default 8).",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Report stats without writing output."
    )
    parser.add_argument(
        "--hull_ratio_max",
        type=float,
        default=3.0,
        help="Skip hand-object check for objects whose convex hull volume is "
        "more than this multiple of the mesh volume (default 3.0).  "
        "Hollow / concave objects (AR glasses=26x, mugs=4.7x, bowls=4.9x) "
        "produce large false positives when the hand reaches inside them; "
        "3.0 keeps solid objects (cans=1.3x, food=1.0x) while skipping "
        "open containers.  Set to 0 to disable all hand-object checks.",
    )
    parser.add_argument(
        "--sequence_id", type=str, default=None, help="Evaluate a single sequence."
    )
    parser.add_argument(
        "--sequence_pattern", type=str, default=None, help="Regex to filter sequences."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.input_dir.exists():
        log.error("input_dir does not exist: %s", args.input_dir)
        sys.exit(1)

    log.info("Parsing collision geometry from URDFs ...")
    right_shapes = _parse_collision_shapes(_URDF_RIGHT)
    left_shapes = _parse_collision_shapes(_URDF_LEFT)
    log.info("  right: %d shapes, left: %d shapes", len(right_shapes), len(left_shapes))

    parquet_dirs = _find_parquet_dirs(args.input_dir)
    log.info("Found %d sequences", len(parquet_dirs))

    # Apply filters
    if args.sequence_id:
        parquet_dirs = [
            d
            for d in parquet_dirs
            if d.parent.name == f"sequence_id={args.sequence_id}"
        ]
    if args.sequence_pattern:
        import re  # noqa: PLC0415

        pat = re.compile(args.sequence_pattern)
        parquet_dirs = [
            d
            for d in parquet_dirs
            if pat.search(d.parent.name.replace("sequence_id=", ""))
        ]

    log.info(
        "Processing %d sequences (stride=%d, max_pen=%.3fm, hull_ratio_max=%.1f) ...",
        len(parquet_dirs),
        args.stride,
        args.max_penetration,
        args.hull_ratio_max,
    )

    hull_cache: dict = {}

    work_items = [
        (
            d,
            right_shapes,
            left_shapes,
            args.max_penetration,
            args.hull_ratio_max,
            args.stride,
            hull_cache,
        )
        for d in parquet_dirs
    ]

    results = []
    if args.num_workers > 1:
        from multiprocessing import Pool  # noqa: PLC0415

        work_items_mp: list[tuple] = [
            (
                d,
                right_shapes,
                left_shapes,
                args.max_penetration,
                args.hull_ratio_max,
                args.stride,
                {},
            )
            for d in parquet_dirs
        ]
        with Pool(processes=args.num_workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(_check_sequence, work_items_mp, chunksize=4)
            ):
                results.append(res)
                if (i + 1) % 100 == 0:
                    n_rej = sum(1 for r in results if r["rejected"])
                    log.info("  %d/%d  rejected=%d", i + 1, len(parquet_dirs), n_rej)
    else:
        for i, item in enumerate(work_items):
            res = _check_sequence(item)
            results.append(res)
            if (i + 1) % 100 == 0:
                n_rej = sum(1 for r in results if r["rejected"])
                log.info("  %d/%d  rejected=%d", i + 1, len(parquet_dirs), n_rej)

    # Summary
    n_total = len(results)
    n_rejected = sum(1 for r in results if r["rejected"])
    n_valid = n_total - n_rejected
    log.info("=" * 60)
    log.info("Total:    %d", n_total)
    log.info("Valid:    %d (%.1f%%)", n_valid, 100 * n_valid / max(n_total, 1))
    log.info("Rejected: %d (%.1f%%)", n_rejected, 100 * n_rejected / max(n_total, 1))

    ho_pens = [r["max_ho_pen"] for r in results]
    hh_pens = [r["max_hh_pen"] for r in results]
    log.info(
        "Max hand-object pen: %.4f m (mean %.4f m)",
        max(ho_pens),
        sum(ho_pens) / len(ho_pens),
    )
    log.info(
        "Max hand-hand   pen: %.4f m (mean %.4f m)",
        max(hh_pens),
        sum(hh_pens) / len(hh_pens),
    )

    if args.dry_run:
        log.info("Dry-run: no output written.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_report(args.output_dir / "penetration_report.csv", results)
        return

    # Copy valid sequences
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Copying %d valid sequences to %s ...", n_valid, args.output_dir)

    copied = 0
    for res in results:
        if res["rejected"]:
            continue
        seq_dir: Path = res["seq_dir"]  # robot_name= dir
        seq_id_dir = seq_dir.parent  # sequence_id= dir
        dest_seq = args.output_dir / seq_id_dir.name
        dest_robot = dest_seq / seq_dir.name
        dest_robot.mkdir(parents=True, exist_ok=True)
        for f in seq_dir.iterdir():
            dest_f = dest_robot / f.name
            if not dest_f.exists():
                shutil.copy2(str(f), str(dest_f))
        copied += 1
        if copied % 200 == 0:
            log.info("  copied %d/%d", copied, n_valid)

    log.info("Done. %d sequences written to %s", copied, args.output_dir)

    # Write report CSV
    report_path = args.output_dir / "penetration_report.csv"
    _write_report(report_path, results)
    log.info("Report: %s", report_path)


def _write_report(path: Path, results: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "object_name",
                "rejected",
                "reason",
                "max_ho_pen_cm",
                "max_hh_pen_cm",
            ],
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["seq_id"]):
            writer.writerow(
                {
                    "sequence_id": r["seq_id"],
                    "object_name": r.get("object_name", ""),
                    "rejected": r["rejected"],
                    "reason": r["reason"],
                    "max_ho_pen_cm": f"{r['max_ho_pen']*100:.3f}",
                    "max_hh_pen_cm": f"{r['max_hh_pen']*100:.3f}",
                }
            )


# ---------------------------------------------------------------------------
# Quality-check protocol (auto-discovered by data_quality_checks/__init__.py)
# ---------------------------------------------------------------------------

# Module-level cache so URDFs are parsed once per process.
_RIGHT_SHAPES: dict[str, CollisionShape] | None = None
_LEFT_SHAPES: dict[str, CollisionShape] | None = None


def _get_shapes() -> tuple[dict[str, CollisionShape], dict[str, CollisionShape]]:
    """Lazy-load and cache the right/left collision shape dicts (process-local)."""
    global _RIGHT_SHAPES, _LEFT_SHAPES  # noqa: PLW0603 — intentional module-level cache
    if _RIGHT_SHAPES is None:
        _RIGHT_SHAPES = _parse_collision_shapes(_URDF_RIGHT)
        _LEFT_SHAPES = _parse_collision_shapes(_URDF_LEFT)
    assert _LEFT_SHAPES is not None  # narrowed by _RIGHT_SHAPES check above
    return _RIGHT_SHAPES, _LEFT_SHAPES


def check(
    data: dict,
    seq_dir: Path | None = None,
    max_penetration: float = 0.02,
    stride: int = 3,
    hull_ratio_max: float = 3.0,
) -> dict:
    """Pass iff max hand-object and hand-hand penetration ≤ max_penetration (default 2 cm).

    score = max penetration depth in centimetres across all sampled frames.
    """
    right_shapes, left_shapes = _get_shapes()

    right_frames_seq = data.get("robot_right_frames", [[]])[0]
    left_frames_seq = data.get("robot_left_frames", [[]])[0]
    right_frame_names = data.get("right_robot_frame_names", [[]])[0]
    left_frame_names = data.get("left_robot_frame_names", [[]])[0]
    obj_positions = data.get("object_body_position", [[]])[0]
    obj_wxyz = data.get("object_body_wxyz", [[]])[0]
    obj_mesh_paths = data.get("object_mesh_paths", [[]])[0]

    n_frames = len(right_frames_seq)
    if n_frames == 0:
        return {"pass": True, "score": 0.0, "reason": "no frames"}

    right_hsc = _HandShapeCache(right_shapes, right_frame_names)
    left_hsc = _HandShapeCache(left_shapes, left_frame_names)

    hull_cache: dict = {}
    hull_entries: list[tuple] = []
    for mp in obj_mesh_paths:
        hull, ratio = _load_hull(mp, hull_cache)
        hull_entries.append((None, ratio) if ratio > hull_ratio_max else (hull, ratio))

    max_ho_pen = 0.0
    max_hh_pen = 0.0

    for t in range(0, n_frames, max(1, stride)):
        right_caps = right_hsc.world_spheres(right_frames_seq[t])
        left_caps = left_hsc.world_spheres(left_frames_seq[t])

        if obj_positions and len(obj_positions) > t:
            for body_idx, (hull, _) in enumerate(hull_entries):
                if hull is None or body_idx >= len(obj_positions[t]):
                    continue
                obj_pos = np.array(obj_positions[t][body_idx], dtype=float)
                obj_R = _quat_wxyz_to_matrix(obj_wxyz[t][body_idx])
                ho = _max_hand_object_penetration(
                    right_caps + left_caps, hull, obj_pos, obj_R
                )
                max_ho_pen = max(max_ho_pen, ho)

        hh = _max_hand_hand_penetration(right_caps, left_caps)
        max_hh_pen = max(max_hh_pen, hh)

        if max_ho_pen > max_penetration and max_hh_pen > max_penetration:
            break

    overall_max = max(max_ho_pen, max_hh_pen)
    passed = overall_max <= max_penetration

    parts = []
    if max_ho_pen > max_penetration:
        parts.append(f"hand_object:{max_ho_pen*100:.1f}cm")
    if max_hh_pen > max_penetration:
        parts.append(f"hand_hand:{max_hh_pen*100:.1f}cm")
    reason = ",".join(parts) if parts else f"ok (max={overall_max*100:.2f}cm)"

    return {"pass": passed, "score": round(overall_max * 100, 4), "reason": reason}


if __name__ == "__main__":
    main()
