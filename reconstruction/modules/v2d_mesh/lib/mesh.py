from dataclasses import dataclass, field
import numpy as np
import trimesh
from PIL import Image as PILImage

from v2d.common.datatypes import BoundingBox3d, Image  # re-exported for convenience


@dataclass
class Mesh:
    """
    Triangle mesh with optional per-vertex colors or UV texture.

    Vertices are in a right-handed coordinate system. The canonical camera
    convention used by mesh_render_* functions is OpenCV: camera at the origin
    looking along +Z, Y axis pointing down.

    Appearance priority (highest to lowest):
      1. uv + texture  — UV-mapped texture atlas (TextureVisuals)
      2. vertex_colors — per-vertex RGBA (ColorVisuals)
      3. neither       — default trimesh appearance
    """
    vertices: np.ndarray                 # (N, 3) float64
    faces: np.ndarray                    # (F, 3) int64
    vertex_colors: np.ndarray | None = None   # (N, 4) uint8 RGBA
    uv: np.ndarray | None = None              # (N, 2) float64 UV coordinates
    texture: np.ndarray | None = None         # (H, W, 4) uint8 RGBA texture image

    def to_trimesh(self) -> trimesh.Trimesh:
        tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        if self.uv is not None and self.texture is not None:
            pil_img = PILImage.fromarray(self.texture)
            material = trimesh.visual.material.SimpleMaterial(image=pil_img)
            tm.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=material)
        elif self.vertex_colors is not None:
            tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=self.vertex_colors)
        return tm

    @staticmethod
    def from_trimesh(tm: trimesh.Trimesh) -> 'Mesh':
        uv = None
        texture = None
        vertex_colors = None

        if isinstance(tm.visual, trimesh.visual.TextureVisuals):
            if tm.visual.uv is not None:
                uv = np.array(tm.visual.uv, dtype=np.float64)
            mat = tm.visual.material
            img = getattr(mat, 'image', None)
            if img is not None:
                texture = np.array(img.convert('RGBA'), dtype=np.uint8)
        elif hasattr(tm.visual, 'vertex_colors'):
            vc = np.array(tm.visual.vertex_colors)
            if vc.shape[0] == len(tm.vertices):
                vertex_colors = vc

        return Mesh(
            vertices=np.array(tm.vertices, dtype=np.float64),
            faces=np.array(tm.faces, dtype=np.int64),
            vertex_colors=vertex_colors,
            uv=uv,
            texture=texture,
        )

    def save(self, path: str) -> None:
        """Export mesh to file. Format is inferred from the extension (e.g. .glb, .obj)."""
        self.to_trimesh().export(path)

    @staticmethod
    def load(path: str) -> 'Mesh':
        """Load a mesh from file. Multi-geometry scenes are merged into one mesh."""
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if len(geometries) == 1:
                tm = geometries[0]
            else:
                tm = trimesh.util.concatenate(geometries)
        else:
            tm = loaded
        return Mesh.from_trimesh(tm)
