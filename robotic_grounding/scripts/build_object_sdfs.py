#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompute per-object SDFs for the hand-object penetration check.

Discovers every unique object mesh under each dataset's
``<hmd>/<dataset>/object_assets/meshes/`` tree and writes a co-located
``<mesh>.sdf.npz`` next to each one. Run once per dataset (offline); the
penetration check then loads these grids and does trilinear lookups instead of
the ~11x-slower convex-hull signed-distance, and gets the true (hollow-aware)
geometry the RL sim actually collides against.

Usage
-----
  python scripts/build_object_sdfs.py --hmd <HMD> --datasets hot3d
  python scripts/build_object_sdfs.py --hmd <HMD>            # all datasets
  python scripts/build_object_sdfs.py --hmd <HMD> --force --jobs 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_penetrations import _resolve_mesh_path  # noqa: E402
from object_sdf import (  # noqa: E402
    DEFAULT_PAD,
    DEFAULT_PITCH,
    build_sdf,
    save_sdf,
    sdf_cache_path,
)

log = logging.getLogger("build_object_sdfs")

# Datasets with object meshes to precompute (registry-supported set).
ALL_DATASETS = ["taco", "hot3d", "arctic", "grab", "h2o", "dexycb", "oakink2"]
MESH_EXTS = (".glb", ".obj", ".ply", ".stl")


def _mesh_roots(
    datasets: list[str], hmd: str | None, object_assets: str | None
) -> list[Path]:
    """The mesh directories to scan, from --object-assets or --hmd."""
    roots = []
    for ds in datasets:
        if object_assets:
            roots.append(Path(object_assets) / "meshes" / ds)
        else:
            assert hmd is not None  # argparse requires one of --hmd/--object-assets
            roots.append(Path(hmd) / ds / "object_assets" / "meshes")
    return roots


def _discover(roots: list[Path]) -> list[Path]:
    """Every unique object mesh under the given mesh directories (folder scan)."""
    meshes: set[Path] = set()
    for root in roots:
        if not root.exists():
            log.info("skip: no %s", root)
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in MESH_EXTS:
                meshes.add(path)
    return sorted(meshes)


def _discover_from_processed(processed_dir: str) -> list[Path]:
    """Unique object meshes referenced across ALL sequences' processed parquet.

    This is the exact set the penetration check will look up (`object_mesh_paths`),
    so it guarantees coverage and skips unreferenced meshes (e.g. visual STLs). Uses
    the check's own `_resolve_mesh_path` so builder and check agree on the file.
    """
    meshes: set[Path] = set()
    for pq_file in Path(processed_dir).rglob("*.parquet"):
        try:
            col = pq.read_table(
                str(pq_file), columns=["object_mesh_paths"]
            ).to_pydict()["object_mesh_paths"]
        except Exception as exc:  # noqa: BLE001 — skip unreadable, keep scanning
            log.warning("read %s: %s", pq_file, exc)
            continue
        for mesh_path in col[0] if col else []:
            local = _resolve_mesh_path(mesh_path, pq_file.parent)
            if os.path.exists(local):
                meshes.add(Path(local))
            else:
                log.warning("mesh not found: %s (from %s)", local, mesh_path)
    return sorted(meshes)


def _worker(task: tuple) -> tuple:
    mesh_path, pitch, pad, force = task
    cache = sdf_cache_path(str(mesh_path))
    if os.path.exists(cache) and not force:
        return (mesh_path, "skip", 0.0, os.path.getsize(cache) / 1e6)
    start = time.time()
    try:
        grid = build_sdf(str(mesh_path), pitch=pitch, pad=pad)
        save_sdf(cache, grid)
        return (mesh_path, "ok", time.time() - start, os.path.getsize(cache) / 1e6)
    except Exception as exc:  # noqa: BLE001 — report and continue over the batch
        return (mesh_path, f"error: {exc}", time.time() - start, 0.0)


def main() -> int:
    """Discover meshes for the requested datasets and build/cache their SDFs."""
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--processed-dir",
        dest="processed_dir",
        help="a {ds}_processed dir; build only meshes its sequences reference "
        "(preferred — exact coverage, no unused meshes)",
    )
    src.add_argument(
        "--hmd", help="human-motion-data root (scans <hmd>/<ds>/object_assets/meshes)"
    )
    src.add_argument(
        "--object-assets",
        dest="object_assets",
        help="object_assets root (scans <object-assets>/meshes/<ds>)",
    )
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--pitch", type=float, default=DEFAULT_PITCH)
    parser.add_argument("--pad", type=float, default=DEFAULT_PAD)
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 1), help="parallel workers"
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if the .sdf.npz exists"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.processed_dir:
        meshes = _discover_from_processed(args.processed_dir)
        src_desc = args.processed_dir
    else:
        roots = _mesh_roots(args.datasets, args.hmd, args.object_assets)
        meshes = _discover(roots)
        src_desc = str(roots)
    if not meshes:
        log.error("No object meshes found under %s", src_desc)
        return 1
    log.info(
        "Building SDFs for %d unique meshes (pitch=%.1fmm, pad=%.0fcm, jobs=%d)",
        len(meshes),
        args.pitch * 1000,
        args.pad * 100,
        args.jobs,
    )

    tasks = [(m, args.pitch, args.pad, args.force) for m in meshes]
    n_ok = n_skip = n_err = 0
    total_mb = 0.0
    build_times: list[float] = []
    start = time.time()
    with Pool(args.jobs) as pool:
        for i, (mesh, status, secs, mb) in enumerate(
            pool.imap_unordered(_worker, tasks), 1
        ):
            total_mb += mb
            if status == "ok":
                n_ok += 1
                build_times.append(secs)
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                log.warning("  [%d/%d] %s -> %s", i, len(meshes), mesh.name, status)
            if i % 25 == 0 or i == len(meshes):
                log.info(
                    "  %d/%d (ok=%d skip=%d err=%d)",
                    i,
                    len(meshes),
                    n_ok,
                    n_skip,
                    n_err,
                )

    med = sorted(build_times)[len(build_times) // 2] if build_times else 0.0
    log.info(
        "Done in %.0fs: %d built, %d skipped, %d errors. Cache %.0f MB total, "
        "median build %.1fs/object.",
        time.time() - start,
        n_ok,
        n_skip,
        n_err,
        total_mb,
        med,
    )
    return 1 if n_err and not n_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
