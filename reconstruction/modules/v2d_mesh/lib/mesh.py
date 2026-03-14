from dataclasses import dataclass
import numpy as np
import trimesh

from v2d.common.datatypes import BoundingBox3d, Image  # re-exported for convenience


@dataclass
class Mesh:
    """
    Triangle mesh with optional per-vertex RGBA colors.

    Vertices are in a right-handed coordinate system. The canonical camera
    convention used by mesh_render_* functions is OpenCV: camera at the origin
    looking along +Z, Y axis pointing down.
    """
    vertices: np.ndarray       # (N, 3) float64
    faces: np.ndarray          # (F, 3) int64
    vertex_colors: np.ndarray | None = None  # (N, 4) uint8 RGBA, optional

    def to_trimesh(self) -> trimesh.Trimesh:
        tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        if self.vertex_colors is not None:
            tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=self.vertex_colors)
        return tm

    @staticmethod
    def from_trimesh(tm: trimesh.Trimesh) -> 'Mesh':
        vertex_colors = None
        if hasattr(tm.visual, 'vertex_colors'):
            vc = np.array(tm.visual.vertex_colors)
            if vc.shape[0] == len(tm.vertices):
                vertex_colors = vc
        return Mesh(
            vertices=np.array(tm.vertices, dtype=np.float64),
            faces=np.array(tm.faces, dtype=np.int64),
            vertex_colors=vertex_colors,
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
