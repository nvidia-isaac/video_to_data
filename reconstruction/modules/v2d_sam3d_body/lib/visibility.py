# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh
from tqdm import tqdm

from v2d.mv.math.numpy_fn import se3_inv, visible_vertices

ZBUF_EPS = 0.005  # 5mm — accounts for rasterization discretization


def _front_facing_mask(
    verts: np.ndarray,
    normals: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Return (V,) bool mask of vertices whose normal faces the camera."""
    cam_pos = T[:3, 3]
    view_dirs = cam_pos - verts  # (V, 3)
    dots = (normals * view_dirs).sum(axis=1)
    return dots > 0


def compute_keypoint_visibility(
    pred_keypoints_3d: np.ndarray,
    pred_vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    image_size: tuple[int, int],
    k_nearest: int = 50,
) -> np.ndarray:
    """Compute per-keypoint visibility as fraction of visible K-nearest vertices.

    For each frame, renders the mesh z-buffer and determines vertex visibility.
    Each keypoint's visibility is the fraction of its K nearest front-facing
    mesh vertices that are visible.

    Args:
        pred_keypoints_3d: (N, P, 3) keypoints in the frame implied by T.
        pred_vertices: (N, V, 3) mesh vertices in the frame implied by T.
        faces: (F, 3) mesh face indices (constant topology).
        K: (3, 3) camera intrinsics.
        T: (4, 4) camera-to-world extrinsic.
        image_size: (W, H) render resolution.
        k_nearest: Number of nearest vertices per keypoint for visibility proxy.

    Returns:
        (N, P) float32 array in [0, 1] — fraction of visible neighbors per keypoint.
    """
    from v2d.mv.vis.renderer import Renderer
    import trimesh

    n_frames, n_kp, _ = pred_keypoints_3d.shape
    n_verts = pred_vertices.shape[1]
    W, H = image_size

    weights = np.zeros((n_frames, n_kp), dtype=np.float32)
    with Renderer(image_size=(W, H)) as renderer:
        for i in tqdm(range(n_frames), desc="Visibility (kNN CPU)"):
            verts_i = pred_vertices[i]
            kps_i = pred_keypoints_3d[i]

            frame_mesh = trimesh.Trimesh(vertices=verts_i, faces=faces, process=False)
            mesh_zbuf = renderer.render_depth([frame_mesh], K, T)
            vert_vis = visible_vertices(verts_i, mesh_zbuf, K, T, zbuf_eps=ZBUF_EPS)

            normals_i = frame_mesh.vertex_normals
            front = _front_facing_mask(verts_i, normals_i, T)
            front_idx = np.where(front)[0]

            if len(front_idx) < k_nearest:
                continue

            tree = cKDTree(verts_i[front_idx])
            _, nn_local = tree.query(kps_i, k=k_nearest)
            nn_global = front_idx[nn_local]  # (P, k_nearest)

            vis_of_neighbors = vert_vis[nn_global]  # (P, k_nearest) bool
            weights[i] = vis_of_neighbors.mean(axis=1).astype(np.float32)

    return weights


def compute_keypoint_visibility_raycast(
    pred_keypoints_3d: np.ndarray,
    pred_vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    max_intersections: int = 1,
) -> np.ndarray:
    """Compute per-keypoint visibility by counting ray-mesh intersections.

    For each keypoint, casts a ray from the camera origin through the keypoint.
    Counts mesh surface intersections before the keypoint's depth. An interior
    keypoint with no occlusion has exactly 1 intersection (entering the body).
    Cross-body occlusion (e.g. arm in front of torso) adds 2+ extra
    intersections. No rendering needed.

    Args:
        pred_keypoints_3d: (N, P, 3) keypoints in the frame implied by T.
        pred_vertices: (N, V, 3) mesh vertices in the frame implied by T.
        faces: (F, 3) mesh face indices (constant topology).
        K: (3, 3) camera intrinsics.
        T: (4, 4) camera-to-world extrinsic.
        max_intersections: Maximum allowed intersections before keypoint
            depth for it to be considered visible. Default 1 (one entry
            into the body mesh is expected for interior keypoints).

    Returns:
        (N, P) float32 array — 1.0 (visible) or 0.0 (occluded).
    """
    n_frames, n_kp, _ = pred_keypoints_3d.shape
    T_inv = se3_inv(T)
    cam_pos = T[:3, 3]

    mesh = trimesh.Trimesh(vertices=pred_vertices[0], faces=faces, process=False)

    weights = np.zeros((n_frames, n_kp), dtype=np.float32)
    for i in tqdm(range(n_frames), desc="Visibility (raycast CPU)"):
        mesh.vertices = pred_vertices[i]
        mesh._cache.clear()

        kps_i = pred_keypoints_3d[i]

        kps_hom = np.concatenate([kps_i, np.ones((n_kp, 1))], axis=1)
        kp_z = (kps_hom @ T_inv.T)[:, 2]  # (P,)

        ray_origins = np.tile(cam_pos, (n_kp, 1))  # (P, 3)
        ray_dirs = kps_i - cam_pos  # (P, 3)
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)

        hit_locations, hit_ray_idx, _ = mesh.ray.intersects_location(
            ray_origins, ray_dirs, multiple_hits=True,
        )

        if len(hit_locations) == 0:
            weights[i] = 1.0
            continue

        hits_hom = np.concatenate([hit_locations, np.ones((len(hit_locations), 1))], axis=1)
        hit_z = (hits_hom @ T_inv.T)[:, 2]

        in_front = hit_z < kp_z[hit_ray_idx] - ZBUF_EPS
        counts = np.zeros(n_kp, dtype=int)
        np.add.at(counts, hit_ray_idx[in_front], 1)

        has_hits = np.zeros(n_kp, dtype=bool)
        has_hits[hit_ray_idx] = True
        weights[i] = np.where(
            has_hits, (counts <= max_intersections).astype(np.float32), 1.0,
        )

    return weights


# ---------------------------------------------------------------------------
# GPU-accelerated visibility functions (PyTorch3D)
# ---------------------------------------------------------------------------


def compute_keypoint_visibility_gpu(
    pred_keypoints_3d: torch.Tensor,
    pred_vertices: torch.Tensor,
    faces: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    image_size: tuple[int, int],
    k_nearest: int = 50,
    batch_size: int = 32,
) -> torch.Tensor:
    """GPU-batched keypoint visibility via KNN + z-buffer.

    GPU-accelerated version of compute_keypoint_visibility. Renders the
    z-buffer in batches, computes vertex normals and front-facing masks
    on GPU, uses torch.cdist for KNN, and averages neighbour visibility.

    Args:
        pred_keypoints_3d: (N, P, 3) keypoints in the frame implied by T.
        pred_vertices: (N, V, 3) mesh vertices in the frame implied by T.
        faces: (F, 3) face indices (int64, constant topology).
        K: (3, 3) camera intrinsics.
        T: (4, 4) world-from-camera extrinsic.
        image_size: (W, H) render resolution.
        k_nearest: Number of nearest front-facing vertices per keypoint.
        batch_size: Max frames per GPU rasterization call.

    Returns:
        (N, P) float32 tensor — fraction of visible neighbours per keypoint.
    """
    import torch
    from .renderer_gpu import GPURenderer

    device = pred_vertices.device
    N, P, _ = pred_keypoints_3d.shape
    V = pred_vertices.shape[1]
    W, H = image_size

    renderer = GPURenderer(image_size, device=device)
    T_inv = torch.inverse(T.float().to(device))
    K_f = K.float().to(device)
    cam_pos = T[:3, 3].float().to(device)

    chunks = []
    for start in tqdm(
        range(0, N, batch_size),
        desc="Visibility (kNN GPU)",
        total=(N + batch_size - 1) // batch_size,
    ):
        end = min(start + batch_size, N)
        B = end - start

        verts_b = pred_vertices[start:end]
        kps_b = pred_keypoints_3d[start:end]

        zbuf = renderer.render_depth(verts_b, faces, K, T)  # (B, H, W)

        # --- vertex visibility via z-buffer comparison ---
        verts_cam = verts_b.float() @ T_inv[:3, :3].T + T_inv[:3, 3]
        vert_z = verts_cam[..., 2]

        uv = verts_cam / vert_z.unsqueeze(-1).clamp(min=1e-6)
        uv = uv @ K_f.T
        u = uv[..., 0].round().long()
        v_coord = uv[..., 1].round().long()

        in_bounds = (u >= 0) & (u < W) & (v_coord >= 0) & (v_coord < H) & (vert_z > 0)
        u_safe = u.clamp(0, W - 1)
        v_safe = v_coord.clamp(0, H - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand(-1, V)
        zbuf_at_vert = zbuf[bi, v_safe, u_safe]

        zbuf_matches = (zbuf_at_vert - vert_z).abs() < ZBUF_EPS
        vert_vis = in_bounds & zbuf_matches & (zbuf_at_vert > 0)

        # --- vertex normals on GPU ---
        fv0 = verts_b[:, faces[:, 0]]
        fv1 = verts_b[:, faces[:, 1]]
        fv2 = verts_b[:, faces[:, 2]]
        face_normals = torch.cross(fv1 - fv0, fv2 - fv0, dim=-1)

        vertex_normals = torch.zeros_like(verts_b)
        for k in range(3):
            idx = faces[:, k].unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)
            vertex_normals.scatter_add_(1, idx, face_normals)
        norms = vertex_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vertex_normals = vertex_normals / norms

        # --- front-facing mask ---
        view_dirs = cam_pos - verts_b.float()
        front = (vertex_normals * view_dirs).sum(dim=-1) > 0

        # --- KNN among front-facing vertices ---
        dists = torch.cdist(kps_b.float(), verts_b.float())  # (B, P, V)
        dists.masked_fill_(~front.unsqueeze(1), float('inf'))
        _, nn_idx = dists.topk(k_nearest, dim=-1, largest=False)

        bi_kp = torch.arange(B, device=device).view(B, 1, 1).expand(-1, P, k_nearest)
        vis_nn = vert_vis[bi_kp, nn_idx]
        chunks.append(vis_nn.float().mean(dim=-1))

    return torch.cat(chunks, dim=0)


def compute_keypoint_visibility_raycast_gpu(
    pred_keypoints_3d: torch.Tensor,
    pred_vertices: torch.Tensor,
    faces: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    image_size: tuple[int, int],
    faces_per_pixel: int = 4,
    max_intersections: int = 1,
    batch_size: int = 8,
    render_scale: float = 0.5,
) -> torch.Tensor:
    """GPU keypoint visibility via depth-peeling (multi-layer z-buffer).

    GPU-accelerated alternative to compute_keypoint_visibility_raycast.
    Uses PyTorch3D's MeshRasterizer with faces_per_pixel > 1 to obtain
    multiple depth layers per pixel, then counts how many mesh surfaces
    lie between the camera and each keypoint.

    Args:
        pred_keypoints_3d: (N, P, 3) keypoints in the frame implied by T.
        pred_vertices: (N, V, 3) mesh vertices in the frame implied by T.
        faces: (F, 3) face indices (int64, constant topology).
        K: (3, 3) camera intrinsics.
        T: (4, 4) world-from-camera extrinsic.
        image_size: (W, H) render resolution.
        faces_per_pixel: Number of depth layers for depth peeling.
        max_intersections: Maximum allowed layers in front of keypoint
            for it to be considered visible. Default 1 (one body-entry
            intersection is expected for interior keypoints).
        batch_size: Max frames per GPU rasterization call.
        render_scale: Scale factor for rasterization resolution (0, 1].
            Lower values save memory/compute at the cost of depth
            accuracy at keypoint locations. Default 0.5 (half resolution).

    Returns:
        (N, P) float32 tensor — 1.0 (visible) or 0.0 (occluded).
    """
    import torch
    from .renderer_gpu import GPURenderer

    device = pred_vertices.device
    N, P, _ = pred_keypoints_3d.shape
    W, H = image_size

    render_W = int(W * render_scale)
    render_H = int(H * render_scale)
    K_render = K.float().to(device).clone()
    K_render[0] *= render_W / W
    K_render[1] *= render_H / H

    renderer = GPURenderer((render_W, render_H), device=device)
    T_inv = torch.inverse(T.float().to(device))

    chunks = []
    for start in tqdm(
        range(0, N, batch_size),
        desc="Visibility (raycast GPU)",
        total=(N + batch_size - 1) // batch_size,
    ):
        end = min(start + batch_size, N)
        B = end - start

        verts_b = pred_vertices[start:end]
        kps_b = pred_keypoints_3d[start:end]

        zbuf = renderer.render_depth_layers(
            verts_b, faces, K_render, T, faces_per_pixel=faces_per_pixel,
        )  # (B, render_H, render_W, faces_per_pixel)

        kps_cam = kps_b.float() @ T_inv[:3, :3].T + T_inv[:3, 3]
        kp_z = kps_cam[..., 2]

        uv = kps_cam / kp_z.unsqueeze(-1).clamp(min=1e-6)
        uv = uv @ K_render.T
        u = uv[..., 0].round().long()
        v_coord = uv[..., 1].round().long()

        in_bounds = (u >= 0) & (u < render_W) & (v_coord >= 0) & (v_coord < render_H) & (kp_z > 0)

        u_safe = u.clamp(0, render_W - 1)
        v_safe = v_coord.clamp(0, render_H - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand(-1, P)

        layers = zbuf[bi, v_safe, u_safe]  # (B, P, faces_per_pixel)

        valid_layers = layers > 0
        in_front = valid_layers & (layers < (kp_z.unsqueeze(-1) - ZBUF_EPS))
        n_before = in_front.sum(dim=-1)

        weights_b = torch.where(
            in_bounds & (n_before <= max_intersections),
            torch.ones(1, device=device),
            torch.zeros(1, device=device),
        )
        chunks.append(weights_b)

    return torch.cat(chunks, dim=0)
