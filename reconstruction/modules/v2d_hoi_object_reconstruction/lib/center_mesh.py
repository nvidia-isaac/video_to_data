# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Center a mesh at the origin by subtracting its bounding box center.

This is required before running FoundationPose — if the mesh is not centered,
FoundationPose's internal tf_to_centered_mesh correction will be applied twice,
causing the tracked pose to jump by the mesh's center offset on the first frame.

Output format:
  - .obj  — for OBJ input: edits vertex positions in the text file directly and
             copies MTL + texture files unchanged. No re-encoding, pixel-perfect texture.
  - .glb  — converts UV texture to vertex colors (lower resolution but self-contained).

Usage:
    python center_mesh.py --input /data/job/textured_mesh.obj \
                          --output /data/job/mesh_input.obj

    python center_mesh.py --input /data/job/mesh_simplified.glb \
                          --output /data/job/mesh_input.glb
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh


def center_mesh_obj(input_path: Path, output_path: Path):
    """OBJ → OBJ: shift only vertex positions; copy MTL + textures verbatim."""
    # First pass: collect all geometric vertices to compute bounding-box center
    vertices = []
    with open(input_path) as f:
        for line in f:
            if line.startswith('v ') and not line.startswith(('vt ', 'vn ')):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    vertices = np.array(vertices)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    print(f"Center before: {center}")
    print(f"Extents (m):   {extents}")
    print(f"Vertices: {len(vertices)}")

    # Second pass: write new OBJ with shifted vertices; all other lines unchanged
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_names = []
    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('v ') and not line.startswith(('vt ', 'vn ')):
                parts = line.split()
                x = float(parts[1]) - center[0]
                y = float(parts[2]) - center[1]
                z = float(parts[3]) - center[2]
                f_out.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
            elif line.startswith('mtllib '):
                mtl_names.append(line.strip().split(' ', 1)[1])
                f_out.write(line)
            else:
                f_out.write(line)

    # Copy MTL files and their referenced textures to the output directory
    input_dir = input_path.parent
    output_dir = output_path.parent
    if input_dir != output_dir:
        for mtl_name in mtl_names:
            mtl_src = input_dir / mtl_name
            if mtl_src.exists():
                shutil.copy2(mtl_src, output_dir / mtl_name)
                with open(mtl_src) as f:
                    for mtl_line in f:
                        parts = mtl_line.strip().split()
                        if parts and parts[0] in ('map_Kd', 'map_Ka', 'map_Ks',
                                                  'map_bump', 'bump', 'map_d'):
                            tex_name = parts[-1]
                            tex_src = input_dir / tex_name
                            if tex_src.exists():
                                shutil.copy2(tex_src, output_dir / tex_name)
                                print(f"Copied texture: {tex_name}")

    center_after = (vertices.min(axis=0) + vertices.max(axis=0)) / 2 - center
    print(f"Center after:  {center_after}")  # should be ~(0, 0, 0)
    print(f"Saved to {output_path}")


def center_mesh(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_ext = output_path.suffix.lower()

    if input_path.suffix.lower() == '.obj' and output_ext == '.obj':
        # Fast path: text-level edit — texture files are never re-encoded
        center_mesh_obj(input_path, output_path)
        return

    # Trimesh path for GLB and other formats
    scene = trimesh.load(str(input_path), force='scene')

    if len(scene.geometry) > 1:
        print(f"Found {len(scene.geometry)} geometries, merging...")
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh = list(scene.geometry.values())[0]

    center = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
    print(f"Center before: {center}")

    mesh.vertices -= center

    center_after = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
    extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    print(f"Center after:  {center_after}")
    print(f"Extents (m):   {extents}")
    print(f"Faces: {len(mesh.faces)}, Vertices: {len(mesh.vertices)}")

    if output_ext != '.obj':
        # GLB uses PBRMaterial which lacks .image; convert to vertex colors
        if not isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
            print("Converting UV/material texture to vertex colors for GLB export...")
            try:
                mesh.visual = mesh.visual.to_color()
                print(f"Vertex colors shape: {mesh.visual.vertex_colors.shape}")
            except Exception as e:
                print(f"Warning: could not convert texture to vertex colors: {e}")

    trimesh.Scene({'mesh': mesh}).export(str(output_path))
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Center mesh at origin for FoundationPose")
    parser.add_argument('--input', required=True, help='Input mesh path (.glb, .obj, ...)')
    parser.add_argument('--output', required=True, help='Output mesh path (.obj or .glb)')
    args = parser.parse_args()
    center_mesh(args.input, args.output)
