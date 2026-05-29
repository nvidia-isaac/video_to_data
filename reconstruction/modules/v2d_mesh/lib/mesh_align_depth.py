# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import numpy as np

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Transform3d
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import mesh_render_depth


def mesh_align_depth(
    mesh: Mesh,
    depth: DepthImage,
    intrinsics: CameraIntrinsics,
) -> Transform3d:
    """Estimate the scale Transform3d needed to align the mesh to a real depth image.

    The mesh is assumed to be already posed in camera space (Z > 0 and facing
    the camera), matching the convention expected by mesh_render_depth.

    Renders the mesh depth, then finds pixels where both the rendered depth and
    the real depth are valid, and fits a scalar s such that:

        s * depth_rendered ≈ depth_real

    The least-squares solution is:

        s = Σ(d_real · d_rendered) / Σ(d_rendered²)

    Returns:
        A scale-only Transform3d with scale=[s, s, s]. Apply this to the original
        mesh (before posing) via mesh_transform to bring it into the real depth scale.

    Raises:
        ValueError: if there are no valid overlapping pixels to fit against.
    """
    rendered = mesh_render_depth(mesh, intrinsics)

    d_rendered = rendered.depth
    d_real = depth.depth

    valid = (d_rendered > 0) & (d_real > 0) & np.isfinite(d_real) & np.isfinite(d_rendered)
    if not np.any(valid):
        raise ValueError("No valid overlapping pixels between rendered mesh and depth image.")

    dr = d_rendered[valid]
    dl = d_real[valid]
    s = float(np.dot(dl, dr) / np.dot(dr, dr))

    return Transform3d(rotation=[1.0, 0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], scale=[s, s, s])
