#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate rigid URDFs for dataset objects.

Reads parquet data or mesh directories to discover unique objects, then
generates a single-link ``_rigid.urdf`` for each one.  Dataset properties
(mesh scale, format, etc.) are read from the dataset registry.

Usage:
    python scripts/generate_rigid_urdfs.py --dataset taco
    python scripts/generate_rigid_urdfs.py --dataset hot3d
    python scripts/generate_rigid_urdfs.py --dataset all
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import trimesh
from robotic_grounding.retarget import (
    ASSETS_DIR as ASSET_DIR,
)
from robotic_grounding.retarget import (
    HUMAN_MOTION_DATA_DIR,
)
from robotic_grounding.retarget.dataset_registry import (
    get_all_dataset_names,
    get_dataset_config,
)
from robotic_grounding.retarget.naming import make_usd_safe

# Repository root (two levels up from scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = ASSET_DIR / "urdfs"


def _relative_meshdir(urdf_path: Path, mesh_file: Path) -> str:
    """Compute the relative meshdir from URDF location to the mesh's parent dir."""
    urdf_dir = urdf_path.parent
    mesh_dir = mesh_file.parent
    try:
        rel = mesh_dir.relative_to(urdf_dir)
    except ValueError:
        rel = Path(
            *([".."] * len(urdf_dir.relative_to(ASSET_DIR).parts))
        ) / mesh_dir.relative_to(ASSET_DIR)
    return str(rel) + "/"


def _stl_output_dir(dataset: str, safe_object_name: str, source_mesh_dir: Path) -> Path:
    """Where the generated visual STL should land.

    Committed datasets (source mesh already under ASSET_DIR) keep the STL
    next to the source. Runtime-only datasets (dexycb, grab — meshes live
    under HUMAN_MOTION_DATA_DIR which is outside ASSET_DIR on OSMO) get a
    stable home under ASSET_DIR/meshes/{dataset}/{safe_name}/ so the URDF's
    meshdir resolves inside the asset tree.
    """
    try:
        source_mesh_dir.relative_to(ASSET_DIR)
        return source_mesh_dir
    except ValueError:
        return ASSET_DIR / "meshes" / dataset / safe_object_name


def _create_visual_mesh(mesh_path: Path, out_path: Path, scale: float) -> bool:
    """Create a clean visual mesh (STL) from the source OBJ.

    Isaac Sim's OBJ parser can't handle Meshlab's interleaved vn/v format.
    STL is universally supported. Follows ketchup_rigid.urdf pattern.
    """
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception as e:
        print(f"  Warning: failed to load mesh {mesh_path}: {e}")
        return False

    mesh.vertices *= scale

    try:
        mesh.export(str(out_path))
    except Exception as e:
        print(f"  Warning: failed to export visual mesh to {out_path}: {e}")
        return False
    return True


def _generate_urdf(
    safe_object_name: str,
    meshdir: str,
    mesh_filename: str,
) -> str:
    """Generate a rigid URDF. Uses the same mesh for visual and collision.

    Mesh is pre-scaled to meters (scale=1).
    """
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" ?>
        <robot name="{safe_object_name}">
           <mujoco>
              <compiler meshdir="{meshdir}" balanceinertia="true" discardvisual="false"/>
           </mujoco>

           <link name="object">
              <inertial>
                 <origin xyz="0 0 0" rpy="0 0 0"/>
                 <mass value="0.3"/>
                 <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
              </inertial>
              <visual>
                 <origin rpy="0 0 0" xyz="0 0 0"/>
                 <geometry>
                    <mesh filename="{meshdir}{mesh_filename}" scale="1 1 1"/>
                 </geometry>
              </visual>
              <collision>
                 <origin rpy="0 0 0" xyz="0 0 0"/>
                 <geometry>
                    <mesh filename="{meshdir}{mesh_filename}" scale="1 1 1"/>
                 </geometry>
              </collision>
           </link>

        </robot>
    """
    )


def _discover_h2o_objects() -> dict[str, tuple[Path, Path]]:
    """Discover unique H2O objects from OBJ mesh files.

    Searches in order (mirrors :func:`_discover_dexycb_objects`):

    1. ``assets/meshes/h2o/{name}/*.obj`` (committed copy, if any).
    2. ``{HUMAN_MOTION_DATA_DIR}/h2o/dataset/object/{name}/*.obj`` (raw
       dataset staged at OSMO runtime via the CSS mount).

    H2O mesh filenames don't always match the folder name (e.g.
    ``spray/lotion_spray.obj``), so we glob for ``*.obj`` and keep the
    first match.  URDFs go to ``assets/urdfs/h2o/{name}_rigid.urdf``.
    """
    canonical_dir = ASSET_DIR / "meshes" / "h2o"
    runtime_dir = HUMAN_MOTION_DATA_DIR / "h2o" / "dataset" / "object"
    urdf_out_dir = URDF_DIR / "h2o"

    sources = [d for d in (canonical_dir, runtime_dir) if d.is_dir()]
    if not sources:
        print(f"No H2O mesh directory at {canonical_dir} or {runtime_dir}")
        return {}

    objects: dict[str, tuple[Path, Path]] = {}
    for base in sources:
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or sub.name in objects:
                continue
            objs = sorted(sub.glob("*.obj"))
            if not objs:
                continue
            objects[sub.name] = (objs[0], urdf_out_dir / f"{sub.name}_rigid.urdf")
    return objects


def _discover_dexycb_objects() -> dict[str, tuple[Path, Path]]:
    """Discover the 21 YCB objects shipped with DexYCB.

    Meshes live at ``{dataset}/models/{name}/textured_simple.obj``.  Also
    accepts a committed copy under ``assets/meshes/dexycb/``.
    URDFs go to ``assets/urdfs/dexycb/{name}_rigid.urdf``.
    """
    canonical_dir = ASSET_DIR / "meshes" / "dexycb"
    runtime_dir = HUMAN_MOTION_DATA_DIR / "dexycb" / "dataset" / "models"
    urdf_out_dir = URDF_DIR / "dexycb"

    sources = [d for d in (canonical_dir, runtime_dir) if d.is_dir()]
    if not sources:
        print(f"No DexYCB mesh directory at {canonical_dir} or {runtime_dir}")
        return {}

    objects: dict[str, tuple[Path, Path]] = {}
    for base in sources:
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or sub.name in objects:
                continue
            # Loader writes ``make_usd_safe(name)_rigid.urdf`` in the parquet;
            # names starting with a digit (YCB ids like "002_…") get an ``obj_`` prefix.
            safe = make_usd_safe(sub.name)
            preferred = sub / "textured_simple.obj"
            if preferred.exists():
                objects[sub.name] = (preferred, urdf_out_dir / f"{safe}_rigid.urdf")
                continue
            objs = sorted(sub.glob("*.obj"))
            if objs:
                objects[sub.name] = (objs[0], urdf_out_dir / f"{safe}_rigid.urdf")
    return objects


def _discover_grab_objects() -> dict[str, tuple[Path, Path]]:
    """Discover unique GRAB objects from the canonical contact meshes.

    GRAB ships its rigid object meshes at
    ``{dataset}/tools/object_meshes/contact_meshes/{name}.ply`` (where
    ``{dataset}`` is the extracted ``grab_dir`` root on OSMO / locally).
    URDFs go to ``assets/urdfs/grab/{name}_rigid.urdf``.
    """
    meshes_dir = (
        HUMAN_MOTION_DATA_DIR
        / "grab"
        / "dataset"
        / "tools"
        / "object_meshes"
        / "contact_meshes"
    )
    urdf_out_dir = URDF_DIR / "grab"
    if not meshes_dir.is_dir():
        print(f"No GRAB contact_meshes directory at {meshes_dir}")
        return {}

    objects: dict[str, tuple[Path, Path]] = {}
    for ply_path in sorted(meshes_dir.glob("*.ply")):
        name = ply_path.stem
        # Skip the body/hand meshes that live in the same directory.
        if name in {"body", "lhand", "rhand"}:
            continue
        objects[name] = (ply_path, urdf_out_dir / f"{name}_rigid.urdf")
    return objects


def _discover_taco_objects() -> dict[str, tuple[Path, Path]]:
    """Discover TACO objects from committed OBJ meshes.

    TACO meshes live at ``assets/meshes/taco/{id}_cm.obj`` (one per object,
    in centimetres — the ``_cm`` suffix is stripped for the URDF name).
    URDFs go to ``assets/urdfs/taco/{id}_rigid.urdf``.
    """
    meshes_dir = ASSET_DIR / "meshes" / "taco"
    urdf_out_dir = URDF_DIR / "taco"
    if not meshes_dir.is_dir():
        print(f"No TACO mesh directory at {meshes_dir}")
        return {}
    objects: dict[str, tuple[Path, Path]] = {}
    for obj_path in sorted(meshes_dir.glob("*_cm.obj")):
        obj_id = obj_path.stem[: -len("_cm")]
        objects[obj_id] = (obj_path, urdf_out_dir / f"{obj_id}_rigid.urdf")
    return objects


def _discover_oakink2_objects() -> dict[str, tuple[Path, Path]]:
    """Discover OakInk2 objects from the canonical ``object_repair`` meshes.

    OakInk2's raw meshes come in two trees — ``object_raw/align_ds/{id}/*``
    uses per-object mesh filenames (``mug.ply``, ``cup.ply``, ``scan.ply``,
    ``model_align.obj``, …) which makes discovery brittle; only a subset
    of IDs have ``scan.ply``.  ``object_repair/align_ds/{id}/model.obj`` is
    the cleaned-up version and is present for every ID the loader uses
    (see ``oakink2_loader._resolve_mesh_path`` — the loader also prefers
    ``object_repair``).  Matching that path keeps the generator and the
    loader aligned: every object the loader records in ``object_urdf_paths``
    gets a URDF here.
    """
    meshes_dir = ASSET_DIR / "meshes" / "oakink2" / "object_repair" / "align_ds"
    urdf_out_dir = URDF_DIR / "oakink2"
    if not meshes_dir.is_dir():
        print(f"No OakInk2 mesh directory at {meshes_dir}")
        return {}
    objects: dict[str, tuple[Path, Path]] = {}
    for sub in sorted(meshes_dir.iterdir()):
        if not sub.is_dir():
            continue
        mesh = sub / "model.obj"
        if not mesh.exists():
            continue
        # Loader stores URDF paths as ``{object_id}_rigid.urdf`` (no
        # make_usd_safe); keep the same convention here.
        objects[sub.name] = (mesh, urdf_out_dir / f"{sub.name}_rigid.urdf")
    return objects


def _discover_hot3d_objects() -> dict[str, tuple[Path, Path]]:
    """Discover unique Hot3D objects from GLB mesh files.

    Hot3D meshes are stored as {uid}.glb in assets/meshes/hot3d/.
    URDFs go to assets/urdfs/hot3d/{uid}_rigid.urdf.
    """
    meshes_dir = ASSET_DIR / "meshes" / "hot3d"
    urdf_out_dir = URDF_DIR / "hot3d"
    files = sorted(meshes_dir.glob("*.glb"))
    if not files:
        print(f"No GLB meshes found in {meshes_dir}")
        return {}

    objects: dict[str, tuple[Path, Path]] = {}
    for glb_path in files:
        uid = glb_path.stem
        urdf_path = urdf_out_dir / f"{uid}_rigid.urdf"
        objects[uid] = (glb_path, urdf_path)

    return objects


def _discover_objects(dataset: str) -> dict[str, tuple[Path, Path]]:
    """Discover unique (object_id -> mesh_path) from committed/raw mesh files.

    Each rigid dataset has its own layout (TACO flat, OakInk2 nested, etc.).
    Arctic is intentionally unsupported — its URDFs are hand-crafted
    articulated models, not regenerable from meshes alone.
    """
    discovery = {
        "taco": _discover_taco_objects,
        "oakink2": _discover_oakink2_objects,
        "hot3d": _discover_hot3d_objects,
        "h2o": _discover_h2o_objects,
        "grab": _discover_grab_objects,
        "dexycb": _discover_dexycb_objects,
    }
    if dataset not in discovery:
        raise ValueError(
            f"Rigid URDF generation not supported for dataset '{dataset}' "
            f"(articulated datasets must ship hand-crafted URDFs)."
        )
    return discovery[dataset]()


def generate_for_dataset(dataset: str, dry_run: bool = False) -> None:
    """Generate rigid URDFs for all objects in a dataset."""
    print(f"\n{'='*60}")
    print(f"Generating rigid URDFs for {dataset}")
    print(f"{'='*60}")

    objects = _discover_objects(dataset)
    if not objects:
        print("No objects found.")
        return

    out_dir = URDF_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # All generated meshes (visual STL + collision OBJ) go next to source meshes
    config = get_dataset_config(dataset)
    scale = config.mesh_vertex_scale
    generated = 0
    skipped = 0

    for obj_id, (mesh_path, urdf_path) in sorted(objects.items()):
        # Clean STL mesh (avoids Meshlab interleaved OBJ that crashes Isaac Sim)
        safe_object_name = make_usd_safe(obj_id)
        mesh_name = f"{safe_object_name}_visual.stl"
        stl_dir = _stl_output_dir(dataset, safe_object_name, mesh_path.parent)
        mesh_out_path = stl_dir / mesh_name
        meshdir = _relative_meshdir(urdf_path, mesh_out_path)

        if dry_run:
            print(f"  DRY-RUN: would write {urdf_path}")
            continue

        stl_dir.mkdir(parents=True, exist_ok=True)
        if not _create_visual_mesh(mesh_path, mesh_out_path, scale):
            print(f"  SKIP {obj_id}: mesh conversion failed")
            skipped += 1
            continue

        urdf_content = _generate_urdf(
            safe_object_name=safe_object_name,
            meshdir=meshdir,
            mesh_filename=mesh_name,
        )
        urdf_path.write_text(urdf_content)
        generated += 1

    print(f"\nGenerated: {generated}, Skipped: {skipped}")
    print(f"Output directory: {out_dir}")


def main() -> None:
    """Generate rigid URDFs for dataset objects."""
    all_names = list(get_all_dataset_names())
    parser = argparse.ArgumentParser(
        description="Generate rigid URDFs for dataset objects."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=all_names + ["all"],
        default="all",
        help="Which dataset to generate URDFs for.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        # Skip datasets whose objects are articulated (e.g. arctic) — their
        # URDFs are hand-crafted and shipped in the repo, not regenerated here.
        datasets = [
            n for n in all_names if not get_dataset_config(n).has_articulated_objects
        ]
    elif get_dataset_config(args.dataset).has_articulated_objects:
        print(
            f"[INFO] {args.dataset} uses articulated URDFs — keeping the "
            "hand-crafted files committed under assets/urdfs/ and exiting."
        )
        return
    else:
        datasets = [args.dataset]
    for ds in datasets:
        generate_for_dataset(ds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
