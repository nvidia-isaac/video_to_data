import numpy as np
from v2d.common.datatypes import Transform3d
from v2d.mesh.lib.mesh import Mesh


def _build_matrix(transform: Transform3d) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from Transform3d (scale → rotate → translate)."""
    w, x, y, z = transform.rotation
    sx, sy, sz = transform.scale
    tx, ty, tz = transform.translation

    R = np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R @ np.diag([sx, sy, sz])
    M[:3, 3] = [tx, ty, tz]
    return M


def mesh_transform(mesh: Mesh, transform: Transform3d) -> Mesh:
    """Apply a Transform3d (scale → rotate → translate) to all mesh vertices."""
    M = _build_matrix(transform)
    verts_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1), dtype=np.float64)])
    new_verts = (M @ verts_h.T).T[:, :3]
    return Mesh(
        vertices=new_verts,
        faces=mesh.faces.copy(),
        vertex_colors=mesh.vertex_colors.copy() if mesh.vertex_colors is not None else None,
    )
