# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.common.datatypes import BoundingBox3d
from v2d.mesh.lib.mesh import Mesh


def mesh_get_bounding_box(mesh: Mesh) -> BoundingBox3d:
    """Return the axis-aligned bounding box of the mesh vertices."""
    v = mesh.vertices
    return BoundingBox3d(
        x0=float(v[:, 0].min()),
        y0=float(v[:, 1].min()),
        z0=float(v[:, 2].min()),
        x1=float(v[:, 0].max()),
        y1=float(v[:, 1].max()),
        z1=float(v[:, 2].max()),
    )
