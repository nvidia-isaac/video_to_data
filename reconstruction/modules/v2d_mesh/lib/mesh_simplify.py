import trimesh
import numpy as np
from v2d.mesh.lib.mesh import Mesh


def mesh_simplify(mesh: Mesh, face_count: int | None = None, factor: float | None = None) -> Mesh:
    """
    Simplify a mesh via quadric decimation, preserving vertex colors when present.

    Args:
        mesh: Input mesh.
        face_count: Target number of faces. Takes priority over factor.
        factor: Reduction factor in (0, 1] (e.g. 0.1 = keep 10% of faces).
                Defaults to 0.1 if neither argument is provided.
    """
    tm = mesh.to_trimesh()
    orig_faces = len(tm.faces)

    if face_count is not None:
        target = max(1, face_count)
    elif factor is not None:
        target = max(1, int(orig_faces * factor))
    else:
        target = max(1, int(orig_faces * 0.1))

    simplified = tm.simplify_quadric_decimation(face_count=target)

    # Re-project vertex colors onto the simplified mesh via nearest-vertex lookup.
    source_colors = None
    if hasattr(tm.visual, 'vertex_colors') and len(tm.visual.vertex_colors) == len(tm.vertices):
        source_colors = np.array(tm.visual.vertex_colors)
    elif hasattr(tm.visual, 'to_color'):
        try:
            source_colors = np.array(tm.visual.to_color().vertex_colors)
        except Exception:
            pass

    if source_colors is not None:
        try:
            _, index = trimesh.proximity.ProximityQuery(tm).vertex(simplified.vertices)
            simplified.visual = trimesh.visual.ColorVisuals(
                mesh=simplified,
                vertex_colors=source_colors[index],
            )
        except Exception:
            pass

    return Mesh.from_trimesh(simplified)
