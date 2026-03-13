import trimesh
import argparse
import os

def simplify_mesh(input_mesh_path, output_mesh_path, face_count=None, factor=None):
    """
    Simplify a GLB/OBJ mesh using quadratic decimation.
    
    Args:
        input_path: Path to the input mesh file
        output_mesh_path: Path to save the simplified mesh
        face_count: Target number of faces
        factor: Reduction factor (e.g., 0.1 for 10% of original faces)
    """
    print(f"Loading mesh from {input_mesh_path}...")
    scene = trimesh.load(input_mesh_path, force='scene')
    
    # FoundationPose expects a single mesh, so we merge if it's a scene
    if len(scene.geometry) > 1:
        print(f"Found {len(scene.geometry)} geometries, merging...")
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
    else:
        mesh = list(scene.geometry.values())[0]
        
    orig_faces = len(mesh.faces)
    orig_verts = len(mesh.vertices)
    print(f"Original mesh: {orig_faces} faces, {orig_verts} vertices")
    
    if face_count:
        target_faces = face_count
    elif factor:
        target_faces = int(orig_faces * factor)
    else:
        # Default to 10% if nothing specified
        target_faces = int(orig_faces * 0.1)
        
    print(f"Simplifying to target {target_faces} faces...")
    
    # Perform simplification
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    
    # Re-project colors if they exist
    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
        print("Re-projecting vertex colors...")
        try:
            # Find the nearest points on the original mesh for each new vertex
            # trimesh.proximity.ProximityQuery(mesh).vertex returns (distance, index)
            _, index = trimesh.proximity.ProximityQuery(mesh).vertex(simplified.vertices)
            # Assign the colors from the nearest original vertices to the new vertices
            simplified.visual = trimesh.visual.ColorVisuals(
                mesh=simplified, 
                vertex_colors=mesh.visual.vertex_colors[index]
            )
        except Exception as e:
            print(f"Failed to re-project vertex colors: {e}")
    elif hasattr(mesh.visual, 'to_color'):
        print("Original mesh has texture/material, converting to vertex colors for simplification...")
        try:
            mesh_with_colors = mesh.copy()
            mesh_with_colors.visual = mesh.visual.to_color()
            _, index = trimesh.proximity.ProximityQuery(mesh_with_colors).vertex(simplified.vertices)
            simplified.visual = trimesh.visual.ColorVisuals(
                mesh=simplified,
                vertex_colors=mesh_with_colors.visual.vertex_colors[index]
            )
        except Exception as e:
            print(f"Failed to convert and re-project colors: {e}")
    
    new_faces = len(simplified.faces)
    new_verts = len(simplified.vertices)
    print(f"Simplified mesh: {new_faces} faces, {new_verts} vertices")
    
    if hasattr(simplified.visual, 'vertex_colors'):
        print(f"Vertex colors shape: {simplified.visual.vertex_colors.shape}")
    
    print(f"Saving to {output_mesh_path}...")
    simplified.export(output_mesh_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify a 3D mesh for FoundationPose")
    parser.add_argument("--input-mesh", required=True, help="Input mesh file (GLB, OBJ, etc.)")
    parser.add_argument("--output-mesh", required=True, help="Output mesh file (default: simplified_<input>)")
    parser.add_argument("--faces", type=int, help="Target face count")
    parser.add_argument("--factor", type=float, help="Reduction factor (0.0 to 1.0)")
    args = parser.parse_args()
    simplify_mesh(args.input_mesh, args.output_mesh, args.faces, args.factor)
