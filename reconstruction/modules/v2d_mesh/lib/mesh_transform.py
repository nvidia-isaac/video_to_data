import numpy as np
from v2d.common.datatypes import Transform3d
from v2d.mesh.lib.mesh import Mesh


def mesh_transform(mesh: Mesh, transform: Transform3d) -> Mesh:
    """Apply a Transform3d (scale → rotate → translate) to all mesh vertices."""
    M = transform.to_matrix()
    verts_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1), dtype=np.float64)])
    new_verts = (M @ verts_h.T).T[:, :3]
    return Mesh(
        vertices=new_verts,
        faces=mesh.faces.copy(),
        vertex_colors=mesh.vertex_colors.copy() if mesh.vertex_colors is not None else None,
    )
