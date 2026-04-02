#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate rigid URDFs for TACO and OakInk2 objects.

Reads parquet data to discover unique objects, then generates a single-link
``_rigid.urdf`` for each one, following the ``ketchup_rigid.urdf`` pattern:
separate visual mesh and simplified collision mesh (~250 verts).

Usage:
    python scripts/generate_rigid_urdfs.py --dataset taco
    python scripts/generate_rigid_urdfs.py --dataset oakink2
    python scripts/generate_rigid_urdfs.py --dataset all
"""

from __future__ import annotations

import argparse
import glob
import textwrap
from pathlib import Path

import pyarrow.parquet as pq
import trimesh
from robotic_grounding.retarget.dataset_loader_base import make_usd_safe

# Repository root (two levels up from scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO_ROOT / "source" / "robotic_grounding" / "robotic_grounding" / "assets"
URDF_DIR = ASSET_DIR / "urdfs"
HUMAN_MOTION_DATA_DIR = ASSET_DIR / "human_motion_data"


def _extract_object_id(mesh_path: str, dataset: str) -> str:
    """Extract a unique object ID from a mesh path."""
    p = Path(mesh_path)
    if dataset == "taco":
        stem = p.stem
        if stem.endswith("_cm"):
            return stem[: -len("_cm")]
        return stem
    else:
        return p.parent.name


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


def _discover_objects(dataset: str) -> dict[str, tuple[Path, Path]]:
    """Discover unique (object_id -> mesh_path) from processed parquet data."""
    if dataset == "taco":
        pattern = str(
            HUMAN_MOTION_DATA_DIR
            / "taco_processed"
            / "sequence_id=*"
            / "robot_name=*"
            / "*.parquet"
        )
    elif dataset == "oakink2":
        pattern = str(
            HUMAN_MOTION_DATA_DIR
            / "oakink2_processed"
            / "sequence_id=*"
            / "robot_name=*"
            / "*.parquet"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    files = glob.glob(pattern)
    if not files:
        print(f"No parquet files found for {dataset}")
        return {}

    objects: dict[str, tuple[Path, Path]] = {}
    for f in files:
        try:
            data = pq.read_table(f).to_pydict()
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
            continue

        mesh_paths = data.get("object_mesh_paths", [[]])[0] or []
        urdf_paths = data.get("object_urdf_paths", [[]])[0] or []

        for mesh_path_str, urdf_path_str in zip(mesh_paths, urdf_paths, strict=False):
            if not mesh_path_str or not urdf_path_str:
                continue
            obj_id = _extract_object_id(mesh_path_str, dataset)
            if obj_id not in objects:
                assert Path(
                    mesh_path_str
                ).exists(), f"Mesh path {mesh_path_str} does not exist"
                objects[obj_id] = (Path(mesh_path_str), Path(urdf_path_str))

    return objects


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
    scale = 0.01 if dataset == "taco" else 1.0
    generated = 0
    skipped = 0

    for obj_id, (mesh_path, urdf_path) in sorted(objects.items()):
        meshdir = _relative_meshdir(urdf_path, mesh_path)

        # Clean STL mesh (avoids Meshlab interleaved OBJ that crashes Isaac Sim)
        safe_object_name = make_usd_safe(obj_id)
        mesh_name = f"{safe_object_name}_visual.stl"
        mesh_out_path = mesh_path.parent / mesh_name

        if dry_run:
            print(f"  DRY-RUN: would write {urdf_path}")
            continue

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
    """Generate rigid URDFs for TACO/OakInk2 objects."""
    parser = argparse.ArgumentParser(
        description="Generate rigid URDFs for TACO/OakInk2 objects."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["taco", "oakink2", "all"],
        default="all",
        help="Which dataset to generate URDFs for.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    args = parser.parse_args()

    datasets = ["taco", "oakink2"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        generate_for_dataset(ds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
