"""
Gaussian initialisation functions for each entity type.

All initialisation is performed once at the start of optimisation.
The coordinate system is depth-space (from frame 0 of the monocular depth model).
"""

import os
import json
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.gsplat.lib.scene import (
    GaussianScene, ENTITY_BACKGROUND, ENTITY_BODY, ENTITY_OBJECT_BASE,
)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def load_mask(masks_dir: str, object_id: int, frame_idx: int) -> Optional[np.ndarray]:
    """Load a SAM2 mask (grayscale PNG) for a given object and frame. Returns bool H×W."""
    from PIL import Image
    path = os.path.join(masks_dir, str(object_id), f"{frame_idx:06d}.png")
    if not os.path.exists(path):
        return None
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127


def load_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Return a single video frame as uint8 RGB (H, W, 3)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth map to 3D points in depth-space.
    Returns (xyz: M×3, pixel_rc: M×2 [row, col]).
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    d = depth.astype(np.float32)

    if valid_mask is not None:
        sel = valid_mask & (d > 0)
    else:
        sel = d > 0

    u_s, v_s, d_s = u[sel], v[sel], d[sel]
    X = (u_s - intrinsics.cx) * d_s / intrinsics.fx
    Y = (v_s - intrinsics.cy) * d_s / intrinsics.fy
    Z = d_s

    xyz = np.stack([X, Y, Z], axis=-1)
    pix_rc = np.stack([v_s.astype(int), u_s.astype(int)], axis=-1)
    return xyz, pix_rc


# --------------------------------------------------------------------------- #
# Per-entity initialisation
# --------------------------------------------------------------------------- #

def init_background(
    depth_path: str,
    intrinsics: CameraIntrinsics,
    frame_rgb: np.ndarray,
    entity_masks: Dict[int, Optional[np.ndarray]],
    max_gaussians: int = 150_000,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialise background Gaussians by unprojecting depth at non-entity pixels.

    entity_masks: {object_id: bool H×W mask} — pixels to EXCLUDE.
    Returns (positions: N×3, colors: N×3).
    """
    depth = DepthImage.load(depth_path).depth
    H, W = depth.shape

    exclude = np.zeros((H, W), dtype=bool)
    for m in entity_masks.values():
        if m is not None:
            # Close gaps within the mask (e.g. SAM2 lower-leg gap where dark pants
            # break the silhouette) then fill interior holes, then dilate.
            # This prevents background Gaussians from being seeded at entity-surface
            # depths in regions the segmentation missed.
            from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation
            m_closed = binary_closing(m, iterations=12)
            m_filled = binary_fill_holes(m_closed)
            exclude |= binary_dilation(m_filled, iterations=3)

    bg_mask = ~exclude

    xyz, pix = depth_to_pointcloud(depth, intrinsics, valid_mask=bg_mask)
    colors = frame_rgb[pix[:, 0], pix[:, 1]].astype(np.float32) / 255.0

    if len(xyz) > max_gaussians:
        idx = np.random.choice(len(xyz), max_gaussians, replace=False)
        xyz, colors = xyz[idx], colors[idx]

    return (
        torch.tensor(xyz, dtype=torch.float32, device=device),
        torch.tensor(colors, dtype=torch.float32, device=device),
    )


def _subdivide_body_mesh(
    verts: torch.Tensor,    # (V, 3) T-pose positions
    faces: np.ndarray,      # (F, 3) int32 triangle indices
    sw: torch.Tensor,       # (V, J) skinning weights
    colors: torch.Tensor,   # (V, 3)
    vertex_ids: torch.Tensor,  # (V,) long
    n: int,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loop-subdivision (midpoint only) for n rounds.
    Each round inserts one vertex per unique edge → ~4× vertex count per round.
    Skinning weights for new vertices are linearly interpolated from their two parents
    (LBS weight vectors sum to 1, so their mean also sums to 1 — no renormalisation needed).
    """
    for _ in range(n):
        V = verts.shape[0]
        edge_to_mid: dict = {}
        mid_v, mid_sw, mid_col, mid_vid = [], [], [], []
        new_faces = []

        def _get_mid(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in edge_to_mid:
                idx = V + len(mid_v)
                edge_to_mid[key] = idx
                mid_v.append((verts[i] + verts[j]) * 0.5)
                mid_sw.append((sw[i] + sw[j]) * 0.5)
                mid_col.append((colors[i] + colors[j]) * 0.5)
                mid_vid.append(vertex_ids[i])   # inherit from first parent
            return edge_to_mid[key]

        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab = _get_mid(a, b)
            bc = _get_mid(b, c)
            ca = _get_mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

        verts      = torch.cat([verts,      torch.stack(mid_v)],  dim=0)
        sw         = torch.cat([sw,         torch.stack(mid_sw)], dim=0)
        colors     = torch.cat([colors,     torch.stack(mid_col)],dim=0)
        vertex_ids = torch.cat([vertex_ids, torch.stack(mid_vid)],dim=0)
        faces      = np.array(new_faces, dtype=np.int32)

    return verts, faces, sw, colors, vertex_ids


def init_body(
    smpl_deformer,
    betas: Optional[torch.Tensor],
    frame_rgb: np.ndarray,
    depth_path: str,
    intrinsics: CameraIntrinsics,
    body_mask: Optional[np.ndarray],
    n_subdivisions: int = 0,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialise body Gaussians at SMPL T-pose vertex positions.

    n_subdivisions: loop-subdivide the SMPL mesh before creating Gaussians.
      0 = ~6890 Gaussians (one per SMPL vertex)
      1 = ~27K Gaussians  (~4× via edge midpoints)
      2 = ~110K Gaussians (~16×)

    Returns (positions: V×3, colors: V×3, skinning_weights: V×J, vertex_ids: V).
    """
    if betas is None:
        betas = torch.zeros(10, device=device)

    # T-pose vertex positions (no grad needed here — we're just sampling initial positions)
    with torch.no_grad():
        v_rest = smpl_deformer.get_rest_vertices(betas.to(device))  # (V, 3)

    V = v_rest.shape[0]

    # Colour body Gaussians from the video frame at projected pixel locations
    colors = _sample_colors_at_vertices(v_rest.cpu().numpy(), intrinsics, frame_rgb)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)

    # Skinning weights from SMPL — stored in probability space here, converted to
    # log-space below. GaussianScene.skinning_weights applies softmax to recover
    # probabilities, so we must store log(w) to get back the original w.
    sw = smpl_deformer.lbs_weights.clone().to(device)  # (V, J) — probabilities

    # Vertex IDs = just [0 .. V-1]
    vertex_ids = torch.arange(V, dtype=torch.long, device=device)

    if n_subdivisions > 0:
        faces = smpl_deformer.body_model.faces  # (F, 3) numpy int32
        # Subdivide in probability space so interpolation stays meaningful
        v_rest, _, sw, colors, vertex_ids = _subdivide_body_mesh(
            v_rest, faces, sw, colors, vertex_ids, n_subdivisions
        )
        print(f"  [init] Body subdivision {n_subdivisions}× → {v_rest.shape[0]} vertices")

    # Convert to log-space: GaussianScene stores _skinning_weights_raw and recovers
    # via softmax(raw). log(p) → softmax → p recovers the original probabilities.
    sw = torch.log(sw.clamp(min=1e-8))

    return v_rest.to(device), colors, sw, vertex_ids


def _sample_colors_at_vertices(
    vertices: np.ndarray,
    intrinsics: CameraIntrinsics,
    frame_rgb: np.ndarray,
) -> np.ndarray:
    """Project 3D vertices to image and sample RGB colors. Returns (V, 3) float32 [0,1]."""
    H, W = frame_rgb.shape[:2]
    Z = vertices[:, 2]
    valid = Z > 0.01
    u = np.where(valid, intrinsics.fx * vertices[:, 0] / (Z + 1e-8) + intrinsics.cx, 0)
    v = np.where(valid, intrinsics.fy * vertices[:, 1] / (Z + 1e-8) + intrinsics.cy, 0)
    u = np.clip(u, 0, W - 1).astype(int)
    v = np.clip(v, 0, H - 1).astype(int)
    colors = frame_rgb[v, u].astype(np.float32) / 255.0  # (V, 3)
    # Fall back to neutral skin tone for out-of-frame vertices
    colors[~valid] = np.array([0.80, 0.65, 0.55], dtype=np.float32)
    return colors


def init_object_from_mesh(
    mesh_path: str,
    depth_path: str,
    intrinsics: CameraIntrinsics,
    object_mask: Optional[np.ndarray],
    transform_path: Optional[str] = None,
    n_gaussians: int = 5000,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialise object Gaussians from a SAM3D mesh, depth-space aligned.

    Alignment process:
      1. Apply the Transform3d (transform_path) to put the mesh in camera/depth space.
         If transform_path is provided (e.g. from FoundationPose or SAM3D), the mesh is
         assumed to already be metric-scale and depth-aligned — no further correction is
         applied.
      2. If no transform is provided, fall back to a coarse centroid + scale alignment
         against the observed depth at the object mask pixels.

    Returns (positions: N×3, colors: N×3).
    """
    import trimesh

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.geometry.values())

    # --- Step 1: apply transform → camera/depth space -------------------------
    vertices_cam = _apply_sam3d_transform(np.array(mesh.vertices, dtype=np.float64), transform_path)

    # Build a transformed copy for sampling purposes
    mesh_cam = mesh.copy()
    mesh_cam.vertices = vertices_cam

    # --- Step 2: depth alignment (only when no transform provided) ------------
    if transform_path is None or not os.path.exists(transform_path):
        translation, scale = _align_mesh_to_depth(vertices_cam, depth_path, intrinsics, object_mask)
        vertices_depth = vertices_cam * scale + translation
    else:
        # Trust the provided transform — it already aligns to depth space
        vertices_depth = vertices_cam

    mesh_cam.vertices = vertices_depth

    # Sample surface points from the aligned mesh
    n_sample = min(n_gaussians, len(mesh_cam.faces) * 3)
    points, face_ids = trimesh.sample.sample_surface(mesh_cam, n_sample)
    points = np.array(points, dtype=np.float32)

    colors = _mesh_sample_colors(mesh, points, face_ids)  # use original mesh for UV/colors

    return (
        torch.tensor(points, dtype=torch.float32, device=device),
        torch.tensor(colors, dtype=torch.float32, device=device),
    )


def _apply_sam3d_transform(vertices: np.ndarray, transform_path: Optional[str]) -> np.ndarray:
    """
    Apply the SAM3D object-to-camera Transform3d to mesh vertices.
    Transform3d convention: rotation=[w,x,y,z], translation=[tx,ty,tz], scale=[sx,sy,sz].
    vertex_camera = R @ (scale * vertex_local) + translation
    Returns (V, 3) float64.
    """
    if transform_path is None or not os.path.exists(transform_path):
        return vertices

    from scipy.spatial.transform import Rotation
    from v2d.common.datatypes import Transform3d

    t3d = Transform3d.load(transform_path)
    w, x, y, z = t3d.rotation                          # [w, x, y, z] stored order
    R = Rotation.from_quat([x, y, z, w]).as_matrix()   # scipy uses [x,y,z,w]
    scale = np.array(t3d.scale, dtype=np.float64)
    transl = np.array(t3d.translation, dtype=np.float64)

    return (R @ (scale * vertices).T).T + transl        # (V, 3)


def _align_mesh_to_depth(
    vertices_cam: np.ndarray,
    depth_path: str,
    intrinsics: CameraIntrinsics,
    object_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Compute (translation, scale) to align a camera-space mesh to the depth map.

    Strategy:
      - Compute 3D centroid of observed depth at object-mask pixels (depth-space).
      - Compute 3D centroid of the mesh in camera space.
      - scale  = stddev(Z_observed) / stddev(Z_mesh_projected) — matches Z spread
      - transl = centroid_depth - scale * centroid_mesh

    Returns:
      translation: (3,) offset to add after scaling
      scale:       scalar multiplier applied before translation
    """
    depth = DepthImage.load(depth_path).depth
    H, W = depth.shape

    if object_mask is None or not object_mask.any():
        return np.zeros(3, dtype=np.float64), 1.0

    # --- Observed 3D points at object mask pixels ---------------------------
    v_idx, u_idx = np.where(object_mask)
    d_obs = depth[v_idx, u_idx].astype(np.float64)
    valid = d_obs > 0
    if valid.sum() < 5:
        return np.zeros(3, dtype=np.float64), 1.0

    v_idx, u_idx, d_obs = v_idx[valid], u_idx[valid], d_obs[valid]
    X_obs = (u_idx - intrinsics.cx) * d_obs / intrinsics.fx
    Y_obs = (v_idx - intrinsics.cy) * d_obs / intrinsics.fy
    Z_obs = d_obs
    pts_obs = np.stack([X_obs, Y_obs, Z_obs], axis=-1)   # (M, 3)
    centroid_obs = pts_obs.mean(axis=0)                   # (3,)

    # --- Mesh centroid in camera space -------------------------------------
    centroid_mesh = vertices_cam.mean(axis=0)             # (3,)

    # --- Scale from Z spread -----------------------------------------------
    z_std_obs  = float(Z_obs.std()) + 1e-8
    z_std_mesh = float(vertices_cam[:, 2].std()) + 1e-8
    scale = z_std_obs / z_std_mesh

    # --- Translation -------------------------------------------------------
    translation = centroid_obs - scale * centroid_mesh    # (3,)

    return translation, scale


def _mesh_sample_colors(mesh, points: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    """Sample colors from mesh surface points. Returns (N, 3) float32 [0,1]."""
    try:
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vc = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            face_verts = mesh.faces[face_ids]  # (N, 3)
            return vc[face_verts].mean(axis=1)
        elif hasattr(mesh.visual, 'to_color'):
            vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32) / 255.0
            face_verts = mesh.faces[face_ids]
            return vc[face_verts].mean(axis=1)
    except Exception:
        pass
    return np.full((len(points), 3), 0.6, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Full scene assembly
# --------------------------------------------------------------------------- #

def build_scene(
    video_path: str,
    depth_folder: str,
    intrinsics: CameraIntrinsics,
    masks_dir: str,
    entity_role_map: Dict[int, str],              # {object_id: "human" | "object"}
    smpl_deformer=None,
    smpl_betas: Optional[torch.Tensor] = None,
    object_mesh_paths: Optional[Dict[int, str]] = None,
    object_transform_paths: Optional[Dict[int, str]] = None,  # {object_id: transform.json}
    max_bg_gaussians: int = 150_000,
    obj_gaussians_per_object: int = 5000,
    initial_opacity_obj: float = 0.05,
    body_subdivisions: int = 0,
    device: str = 'cuda',
) -> GaussianScene:
    """
    Build a GaussianScene by initialising one entity at a time from frame 0.

    entity_role_map:        maps SAM2 object_id → entity role string
    object_mesh_paths:      maps SAM2 object_id → .obj mesh path (from SAM3D)
    object_transform_paths: maps SAM2 object_id → Transform3d JSON (from SAM3D),
                            used to place the mesh in camera space before depth alignment
    """
    frame0_rgb = load_video_frame(video_path, 0)
    depth_path0 = os.path.join(depth_folder, '000000.png')

    # Collect masks for all entities at frame 0
    frame0_masks: Dict[int, Optional[np.ndarray]] = {}
    for oid in entity_role_map:
        frame0_masks[oid] = load_mask(masks_dir, oid, 0)

    # ------------------------------------------------------------------ #
    # 1. Background
    # ------------------------------------------------------------------ #
    bg_positions, bg_colors = init_background(
        depth_path0, intrinsics, frame0_rgb,
        entity_masks=frame0_masks,
        max_gaussians=max_bg_gaussians,
        device=device,
    )
    N_bg = bg_positions.shape[0]
    bg_entity_ids = torch.full((N_bg,), ENTITY_BACKGROUND, dtype=torch.int32, device=device)

    all_positions = [bg_positions]
    all_colors = [bg_colors]
    all_entity_ids = [bg_entity_ids]
    body_sw = None
    body_vids = None

    # ------------------------------------------------------------------ #
    # 2. Body entities
    # ------------------------------------------------------------------ #
    for oid, role in entity_role_map.items():
        if role != 'human':
            continue
        if smpl_deformer is None:
            print(f"  [init] No SMPL deformer — skipping body entity (object_id={oid})")
            continue

        body_mask_img = frame0_masks.get(oid)
        bp, bc, bsw, bvids = init_body(
            smpl_deformer, smpl_betas, frame0_rgb, depth_path0, intrinsics,
            body_mask_img, n_subdivisions=body_subdivisions, device=device,
        )
        N_body = bp.shape[0]
        body_entity_ids = torch.full((N_body,), ENTITY_BODY, dtype=torch.int32, device=device)

        all_positions.append(bp)
        all_colors.append(bc)
        all_entity_ids.append(body_entity_ids)
        body_sw = bsw
        body_vids = bvids
        print(f"  [init] Body: {N_body} Gaussians (object_id={oid})")

    # ------------------------------------------------------------------ #
    # 3. Object entities
    # ------------------------------------------------------------------ #
    rigid_body_id = 0
    for oid, role in entity_role_map.items():
        if role != 'object':
            continue

        entity_id = ENTITY_OBJECT_BASE + rigid_body_id
        obj_mask_img = frame0_masks.get(oid)

        mesh_path = (object_mesh_paths or {}).get(oid)
        transform_path = (object_transform_paths or {}).get(oid)
        if mesh_path and os.path.exists(mesh_path):
            op, oc = init_object_from_mesh(
                mesh_path, depth_path0, intrinsics, obj_mask_img,
                transform_path=transform_path,
                n_gaussians=obj_gaussians_per_object, device=device,
            )
        else:
            # Fall back to unprojecting depth in object mask region
            print(f"  [init] No mesh for object_id={oid} — using depth unproject fallback")
            if obj_mask_img is not None:
                op, oc = _init_object_from_depth(
                    depth_path0, intrinsics, frame0_rgb, obj_mask_img,
                    max_gaussians=obj_gaussians_per_object, device=device,
                )
            else:
                print(f"  [init] No mask for object_id={oid} — skipping")
                rigid_body_id += 1
                continue

        N_obj = op.shape[0]
        obj_entity_ids = torch.full((N_obj,), entity_id, dtype=torch.int32, device=device)
        all_positions.append(op)
        all_colors.append(oc)
        all_entity_ids.append(obj_entity_ids)
        print(f"  [init] Object {rigid_body_id}: {N_obj} Gaussians (object_id={oid})")
        rigid_body_id += 1

    # ------------------------------------------------------------------ #
    # Assemble
    # ------------------------------------------------------------------ #
    positions = torch.cat(all_positions, dim=0)
    colors = torch.cat(all_colors, dim=0)
    entity_ids = torch.cat(all_entity_ids, dim=0)

    scene = GaussianScene(positions, colors, entity_ids, body_sw, body_vids)

    # Set initial opacity for object Gaussians (default scene starts at ~0.05 for all)
    import math
    opacity_raw = math.log(initial_opacity_obj / (1.0 - initial_opacity_obj))
    with torch.no_grad():
        for rid in range(scene.n_objects()):
            obj_mask = scene.object_mask(rid)
            if obj_mask.any():
                scene._opacities_raw.data[obj_mask] = opacity_raw

    # Store initial canonical positions as a flat anchor tensor parallel to _positions.
    # The anchor loss pulls object Gaussians back toward these fixed reference points
    # to prevent canonical drift. The flat layout survives densification (clone/split/prune)
    # because each new Gaussian inherits its parent's anchor entry.
    if scene.n_objects() > 0:
        scene._anchor_positions = scene._positions.detach().clone()

    print(f"  [init] Total Gaussians: {scene.num_gaussians}")
    return scene


def _init_object_from_depth(
    depth_path: str,
    intrinsics: CameraIntrinsics,
    frame_rgb: np.ndarray,
    object_mask: np.ndarray,
    max_gaussians: int = 5000,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth = DepthImage.load(depth_path).depth
    xyz, pix = depth_to_pointcloud(depth, intrinsics, valid_mask=object_mask)
    colors = frame_rgb[pix[:, 0], pix[:, 1]].astype(np.float32) / 255.0
    if len(xyz) > max_gaussians:
        idx = np.random.choice(len(xyz), max_gaussians, replace=False)
        xyz, colors = xyz[idx], colors[idx]
    return (
        torch.tensor(xyz, dtype=torch.float32, device=device),
        torch.tensor(colors, dtype=torch.float32, device=device),
    )
