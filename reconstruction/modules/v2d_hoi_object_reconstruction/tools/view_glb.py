#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Simple GLB/Mesh Viewer using trimesh

Usage:
    python view_glb.py /path/to/mesh.glb
    python view_glb.py /path/to/mesh.ply
"""

import argparse
import sys
from pathlib import Path

try:
    import trimesh
except ImportError:
    print("❌ trimesh not installed!")
    print("Install with: pip install trimesh")
    sys.exit(1)

# Check for pyglet
try:
    import pyglet
    PYGLET_VERSION = pyglet.version
    PYGLET_AVAILABLE = True
except ImportError:
    PYGLET_VERSION = None
    PYGLET_AVAILABLE = False

if PYGLET_AVAILABLE:
    import trimesh.viewer

    def _apply_flat_color(scene):
        """Replace all geometry visuals with a flat blue-gray color."""
        import numpy as np
        for geom in scene.geometry.values():
            if not hasattr(geom, "faces"):
                continue
            n_faces = len(geom.faces)
            face_colors = np.tile([180, 190, 210, 255], (n_faces, 1)).astype(np.uint8)
            geom.visual = trimesh.visual.ColorVisuals(mesh=geom, face_colors=face_colors)

    class TextureToggleViewer(trimesh.viewer.SceneViewer):
        """SceneViewer with 't' shortcut to toggle texture on/off."""

        def __init__(self, scene, no_texture=False, **kwargs):
            self._original_visuals = {
                name: geom.visual.copy()
                for name, geom in scene.geometry.items()
                if hasattr(geom, "visual")
            }
            self._texture_on = not no_texture
            if no_texture:
                _apply_flat_color(scene)
                kwargs.setdefault("smooth", False)
            super().__init__(scene, **kwargs)

        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.T:
                self._texture_on = not self._texture_on
                for name, geom in self.scene.geometry.items():
                    if not hasattr(geom, "visual"):
                        continue
                    if self._texture_on and name in self._original_visuals:
                        geom.visual = self._original_visuals[name].copy()
                    else:
                        _apply_flat_color(self.scene)
                # delete cached vertex lists to force full recreation with correct smooth setting
                for name in list(self.vertex_list.keys()):
                    self.vertex_list[name].delete()
                self.vertex_list.clear()
                self.vertex_list_hash.clear()
                self.vertex_list_mode.clear()
                self._smooth = self._texture_on  # smooth on with texture, off without
                self._update_vertex_list()
                print(f"[t] texture {'ON' if self._texture_on else 'OFF'}")
            else:
                super().on_key_press(symbol, modifiers)


def view_mesh(mesh_path: Path, export_only: bool = False, show_axes: bool = False,
              no_texture: bool = False) -> None:
    """Load and visualize a mesh file.

    Args:
        mesh_path: Path to the mesh file
        export_only: If True, export snapshots instead of opening viewer
        show_axes: If True, show coordinate axes
        no_texture: If True, show flat geometry color instead of texture
    """
    
    print(f"\n📂 Loading mesh: {mesh_path}")
    
    try:
        # Load mesh (supports GLB, PLY, OBJ, STL, etc.)
        mesh = trimesh.load(mesh_path)
        
        # Fix face winding so normals point outward consistently.
        # GLB meshes from some reconstructions have ~50% inward-facing normals;
        # backface culling in the viewer makes those faces invisible → looks
        # transparent. fix_normals() repairs the winding order.
        if isinstance(mesh, trimesh.Scene):
            for geom in mesh.geometry.values():
                if hasattr(geom, 'faces') and len(geom.faces) > 0:
                    trimesh.repair.fix_normals(geom, multibody=True)

        # Print mesh info
        print(f"\n📊 Mesh Information:")
        print(f"   Type: {type(mesh).__name__}")
        
        # Check if it's a point cloud (no faces)
        is_point_cloud = False
        if isinstance(mesh, trimesh.Scene):
            # GLB files are often loaded as Scene
            print(f"   Scene with {len(mesh.geometry)} geometries")
            for name, geom in mesh.geometry.items():
                if hasattr(geom, 'vertices'):
                    n_faces = len(geom.faces) if hasattr(geom, 'faces') else 0
                    print(f"   - {name}: {len(geom.vertices)} vertices, {n_faces} faces")
                    if n_faces == 0:
                        is_point_cloud = True
        elif hasattr(mesh, 'vertices'):
            # Single mesh or point cloud
            print(f"   Vertices: {len(mesh.vertices)}")
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                print(f"   Faces: {len(mesh.faces)}")
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                    print(f"   Has UV coordinates: Yes")
                else:
                    print(f"   Has UV coordinates: No")
            else:
                print(f"   ⚠️  Point cloud (no faces)")
                is_point_cloud = True
                # Convert point cloud to viewable spheres
                if hasattr(mesh, 'vertices'):
                    import numpy as np
                    points = np.asarray(mesh.vertices)
                    # Create tiny spheres for each point
                    point_size = (points.max() - points.min()) * 0.002
                    spheres = []
                    # Subsample if too many points
                    if len(points) > 10000:
                        indices = np.random.choice(len(points), 10000, replace=False)
                        points = points[indices]
                        print(f"   (Subsampled to 10000 points for visualization)")
                    
                    for i, point in enumerate(points):
                        if i % 1000 == 0:
                            print(f"   Creating visualization... {i}/{len(points)}", end='\r')
                        sphere = trimesh.creation.icosphere(radius=point_size, subdivisions=1)
                        sphere.apply_translation(point)
                        spheres.append(sphere)
                    print(f"   Creating visualization... Done!              ")
                    mesh = trimesh.util.concatenate(spheres)
                    print(f"   ✅ Converted to viewable mesh")
        
        # Add coordinate axes if requested
        if show_axes:
            # Get mesh bounds to scale axes appropriately
            if isinstance(mesh, trimesh.Scene):
                bounds = mesh.bounds
            else:
                bounds = mesh.bounds
            
            # Calculate object center
            center = (bounds[0] + bounds[1]) / 2.0
            
            # Calculate axis length (about 30% of max dimension)
            max_dim = (bounds[1] - bounds[0]).max()
            axis_length = max_dim * 0.3
            axis_radius = axis_length * 0.01
            
            # Create axis arrows at object center
            # X axis - RED
            x_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
            x_axis.apply_transform(trimesh.transformations.rotation_matrix(
                angle=3.14159/2, direction=[0, 1, 0]))
            x_axis.apply_translation([axis_length/2, 0, 0])
            x_axis.apply_translation(center)  # Move to object center
            x_axis.visual.vertex_colors = [255, 0, 0, 255]  # Red
            
            # Y axis - GREEN
            y_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
            y_axis.apply_transform(trimesh.transformations.rotation_matrix(
                angle=-3.14159/2, direction=[1, 0, 0]))
            y_axis.apply_translation([0, axis_length/2, 0])
            y_axis.apply_translation(center)  # Move to object center
            y_axis.visual.vertex_colors = [0, 255, 0, 255]  # Green
            
            # Z axis - BLUE
            z_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
            z_axis.apply_translation([0, 0, axis_length/2])
            z_axis.apply_translation(center)  # Move to object center
            z_axis.visual.vertex_colors = [0, 0, 255, 255]  # Blue
            
            # Add to scene
            if isinstance(mesh, trimesh.Scene):
                mesh.add_geometry(x_axis, node_name='X_axis')
                mesh.add_geometry(y_axis, node_name='Y_axis')
                mesh.add_geometry(z_axis, node_name='Z_axis')
            else:
                # Convert to scene and add axes
                scene = trimesh.Scene([mesh, x_axis, y_axis, z_axis])
                mesh = scene
            
            print(f"\n📐 Coordinate Axes Added:")
            print(f"   Origin at object center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            print(f"   🔴 RED   = +X axis")
            print(f"   🟢 GREEN = +Y axis")
            print(f"   🔵 BLUE  = +Z axis")
            print(f"   Axis length: {axis_length:.3f} units")
        
        # Show mesh in viewer or export snapshot
        if export_only or not PYGLET_AVAILABLE:
            if export_only:
                print(f"\n💡 Export mode: Creating snapshots...")
            else:
                print(f"\n⚠️  Windowed viewer not available (pyglet not installed)")
                print(f"   Install with: pip install 'pyglet<2'")
                print(f"\n💡 Exporting snapshots instead...")
            
            # Export a PNG snapshot
            output_path = mesh_path.parent / f"{mesh_path.stem}_snapshot.png"
            scene = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
            
            try:
                # Render to PNG
                png_data = scene.save_image(resolution=(1920, 1080))
                with open(output_path, 'wb') as f:
                    f.write(png_data)
                print(f"   ✅ Saved snapshot: {output_path}")
            except Exception as e:
                print(f"   ❌ Failed to save snapshot: {e}")
            
            # Export to HTML for browser viewing
            html_path = mesh_path.parent / f"{mesh_path.stem}_viewer.html"
            try:
                scene_html = scene.to_html()
                with open(html_path, 'w') as f:
                    f.write(scene_html)
                print(f"   ✅ Saved HTML viewer: {html_path}")
                print(f"   💡 Open in browser: file://{html_path.absolute()}")
            except Exception as e:
                print(f"   ⚠️  HTML export not available: {e}")
                
        else:
            print(f"\n🎨 Opening viewer (pyglet {PYGLET_VERSION})...")
            print(f"   Controls:")
            print(f"   - Left click + drag: Rotate")
            print(f"   - Right click + drag: Pan")
            print(f"   - Scroll: Zoom")
            print(f"   - 'z': Reset view")
            print(f"   - 'w': Toggle wireframe")
            print(f"   - 'c': Toggle backface culling")
            print(f"   - 'a': Toggle axis")
            print(f"   - 't': Toggle texture (geometry-only view)")
            print(f"   - 'q' or ESC: Quit")
            
            try:
                scene = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
                TextureToggleViewer(scene, no_texture=no_texture, start_loop=True)
            except Exception as e:
                print(f"\n⚠️  Viewer failed: {e}")
                print(f"   Try: pip install 'pyglet<2'")
        
    except Exception as e:
        print(f"\n❌ Error loading mesh: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="View GLB/mesh files using trimesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_glb.py output/skinny_wood_chair_sam3d.glb
  python view_glb.py output/skinny_wood_chair_sam3d.ply
  python view_glb.py mesh.glb --axes  # Show coordinate axes

Coordinate Axes (use --axes to show):
  🔴 RED   = +X axis (right)
  🟢 GREEN = +Y axis (up in Y-up, down in camera coords)
  🔵 BLUE  = +Z axis (toward viewer in Y-up, forward in camera coords)
        """
    )
    
    parser.add_argument("mesh_file", type=Path,
                        help="Path to mesh file (GLB, PLY, OBJ, STL, etc.)")
    parser.add_argument("--export", action="store_true",
                        help="Export snapshot/HTML instead of opening viewer")
    parser.add_argument("--axes", action="store_true",
                        help="Show coordinate axes (disabled by default)")
    parser.add_argument("--no-texture", action="store_true",
                        help="Show flat geometry color instead of texture")

    args = parser.parse_args()

    if not args.mesh_file.exists():
        print(f"❌ File not found: {args.mesh_file}")
        return 1

    view_mesh(args.mesh_file, export_only=args.export, show_axes=args.axes,
              no_texture=args.no_texture)
    return 0


if __name__ == "__main__":
    sys.exit(main())

