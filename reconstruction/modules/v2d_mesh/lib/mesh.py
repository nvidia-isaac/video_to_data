from dataclasses import dataclass, field
import numpy as np
import trimesh
from PIL import Image as PILImage

from v2d.common.datatypes import BoundingBox3d, Image  # re-exported for convenience


@dataclass
class Mesh:
    """
    Triangle mesh with optional texture (UV + image) or per-vertex RGBA colors.

    UV texture takes priority over vertex_colors when both are present.
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
        visual = tm.visual
        if isinstance(visual, trimesh.visual.TextureVisuals):
            try:
                if visual.uv is not None and visual.material is not None:
                    mat = visual.material
                    img = (
                        getattr(mat, 'baseColorTexture', None)
                        or getattr(mat, 'image', None)
                    )
                    if img is not None:
                        uv = np.array(visual.uv, dtype=np.float64)
                        texture = np.array(img.convert('RGBA'), dtype=np.uint8)
            except Exception:
                pass
            if uv is None or texture is None:
                # Fall back to baking if UV/texture extraction failed
                try:
                    visual = visual.to_color()
                except Exception:
                    pass
        if uv is None and hasattr(visual, 'vertex_colors'):
            vc = np.array(visual.vertex_colors)
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
    def load(path: str, *, force_mesh: bool = False) -> 'Mesh':
        """Load a mesh from file.

        By default this uses the historical manual scene merge path: scene
        graph transforms are applied, multi-geometry scenes are merged, and
        multi-geometry textures are baked to vertex colors before merging.
        Pass ``force_mesh=True`` to use trimesh's ``force="mesh"`` coercion
        path, which is useful for GLBs whose scene/texture handling works
        better through trimesh's native flattening.
        """
        if force_mesh:
            tm = trimesh.load(path, process=False, force='mesh')
        else:
            loaded = trimesh.load(path)
            if isinstance(loaded, trimesh.Scene):
                meshes = loaded.dump(concatenate=False)
                if len(meshes) == 1:
                    tm = meshes[0]
                else:
                    baked = []
                    for m in meshes:
                        if isinstance(m.visual, trimesh.visual.TextureVisuals):
                            m.visual = m.visual.to_color()
                        baked.append(m)
                    tm = trimesh.util.concatenate(baked)
            else:
                tm = loaded
        return Mesh.from_trimesh(tm)
