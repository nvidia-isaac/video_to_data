import trimesh
import json
import numpy as np
import argparse
import os

def wxyz_quat_to_rotation_matrix(w, x, y, z):
    rotation_matrix = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
    return rotation_matrix

def transform_mesh(input_mesh_path, output_mesh_path, transform_path):
    """
    Applies ONLY the scaling from a SAM3D transform JSON to a mesh.
    """
    print(f"Loading mesh from {input_mesh_path}...")
    scene = trimesh.load(input_mesh_path, process=False)
    
    # Merge if it's a scene
    if hasattr(scene, 'geometry') and len(scene.geometry) > 0:
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
    else:
        mesh = scene

    print(f"Loading transform from {transform_path}...")
    with open(transform_path, "r") as f:
        transform = json.load(f)

    # Extract scale component
    object_scale = np.array(transform['scale'])

    # 1. Apply ONLY Scale
    print(f"Applying scale: {object_scale}")
    mesh.apply_scale(object_scale)

    print(f"Saving scaled mesh to {output_mesh_path}...")
    mesh.export(output_mesh_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply ONLY scaling to a mesh using SAM3D transform JSON")
    parser.add_argument("--input-mesh", required=True, help="Path to input mesh (GLB/OBJ)")
    parser.add_argument("--output-mesh", required=True, help="Path to save the scaled mesh")
    parser.add_argument("--transform", required=True, help="Path to transform JSON")
    
    args = parser.parse_args()
    
    transform_mesh(args.input_mesh, args.output_mesh, args.transform)

