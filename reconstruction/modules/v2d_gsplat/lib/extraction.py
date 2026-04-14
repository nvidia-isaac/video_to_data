"""
Output extraction: save optimised Gaussians, renders video, refined SMPL params.
"""

import os
import json
import struct
import numpy as np
import torch
from typing import Optional, Dict

from v2d.gsplat.lib.scene import GaussianScene, ENTITY_BACKGROUND, ENTITY_BODY, ENTITY_OBJECT_BASE
from v2d.gsplat.lib.deformation import BodyPoseParams, ObjectPoseParams, rotation_6d_to_matrix


def save_gaussians_ply(scene: GaussianScene, path: str) -> None:
    """
    Save Gaussians to a .ply file in the standard 3DGS format.
    Includes entity_id as a scalar property so it can be loaded back.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with torch.no_grad():
        positions = scene.positions.cpu().numpy().astype(np.float32)      # (N, 3)
        rotations = scene.rotations.cpu().numpy().astype(np.float32)      # (N, 4) wxyz
        scales = torch.log(scene.scales).cpu().numpy().astype(np.float32) # (N, 3) log-scale
        opacities = torch.logit(scene.opacities.clamp(1e-6, 1-1e-6)).cpu().numpy().astype(np.float32)  # (N,) raw
        sh_features = scene.sh_features.cpu().numpy().astype(np.float32)  # (N, 16, 3)
        entity_ids = scene.entity_ids.cpu().numpy().astype(np.int32)      # (N,)

    N = positions.shape[0]

    # Build per-vertex data record
    # PLY format: x y z | nx ny nz (unused) | f_dc_0 f_dc_1 f_dc_2 | f_rest_0..44 |
    #             opacity | scale_0 scale_1 scale_2 | rot_0..3 | entity_id
    sh_dc = sh_features[:, 0, :]          # (N, 3)
    sh_rest = sh_features[:, 1:, :]       # (N, 15, 3)
    sh_rest_flat = sh_rest.reshape(N, -1) # (N, 45)

    # PLY header
    ply_properties = (
        ['x', 'y', 'z']
        + ['nx', 'ny', 'nz']
        + [f'f_dc_{i}' for i in range(3)]
        + [f'f_rest_{i}' for i in range(45)]
        + ['opacity']
        + [f'scale_{i}' for i in range(3)]
        + [f'rot_{i}' for i in range(4)]
        + ['entity_id']
    )

    with open(path, 'wb') as f:
        # ASCII header
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {N}\n"
        )
        for prop in ply_properties:
            dtype = 'int' if prop == 'entity_id' else 'float'
            header += f"property {dtype} {prop}\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))

        # Binary data — pack as float32 row-by-row
        zeros3 = np.zeros((N, 3), dtype=np.float32)
        data = np.concatenate([
            positions,                              # 3
            zeros3,                                 # 3 (normals, unused)
            sh_dc,                                  # 3
            sh_rest_flat,                           # 45
            opacities[:, None],                     # 1
            scales,                                 # 3
            rotations,                              # 4
            entity_ids[:, None].astype(np.float32), # 1 (stored as float for uniform packing)
        ], axis=1)  # (N, 63)

        f.write(data.tobytes())

    print(f"  [ply] Saved {N} Gaussians → {path}")


def save_entities_json(entity_role_map: Dict[int, str], object_mesh_paths: Optional[Dict[int, str]], path: str) -> None:
    """Save entity metadata (role map and mesh paths) to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    info = {
        'entity_role_map': {str(k): v for k, v in entity_role_map.items()},
        'object_mesh_paths': {str(k): v for k, v in (object_mesh_paths or {}).items()},
        'entity_id_mapping': {
            'background': ENTITY_BACKGROUND,
            'body': ENTITY_BODY,
            'object_base': ENTITY_OBJECT_BASE,
        },
    }
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)


def save_object_poses(
    obj_pose_params: ObjectPoseParams,
    output_dir: str,
) -> None:
    """
    Save per-frame SE(3) transforms for each rigid object as an NPZ file.

    The .ply stores canonical Gaussians (no trajectory).  This file carries the
    per-frame pose trajectory that, combined with gaussians.ply, fully describes
    the 4D scene for each object entity.

    Output: {output_dir}/object_poses.npz
      rotations:    (T, K, 3, 3)  rotation matrices
      translations: (T, K, 3)     translations in depth-space units
      T:            int            number of frames
      K:            int            number of objects
    """
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        T, K = obj_pose_params.rotations_6d.shape[:2]
        r6d = obj_pose_params.rotations_6d.view(T * K, 6)
        R = rotation_6d_to_matrix(r6d).view(T, K, 3, 3).cpu().numpy()  # (T, K, 3, 3)
        t = obj_pose_params.translations.cpu().numpy()                   # (T, K, 3)

    out_path = os.path.join(output_dir, 'object_poses.npz')
    np.savez(out_path, rotations=R, translations=t, T=T, K=K)
    print(f"  [poses] Saved object trajectory ({T} frames, {K} objects) → {out_path}")


def save_smpl_results(
    body_pose_params: BodyPoseParams,
    output_dir: str,
    model_type: str = 'smpl',
    gender: str = 'neutral',
) -> None:
    """Save refined SMPL parameters as an NlfResult-compatible NPZ file."""
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        T = body_pose_params.global_orient.shape[0]
        # global_orient is stored as 6D rotation; convert back to axis-angle for NPZ output
        from v2d.gsplat.lib.deformation import rotation_6d_to_matrix, _rotation_matrix_to_axis_angle
        go_R = rotation_6d_to_matrix(body_pose_params.global_orient)  # (T, 3, 3)
        go = _rotation_matrix_to_axis_angle(go_R).cpu().numpy()       # (T, 3)
        bp = body_pose_params.body_pose.cpu().numpy()        # (T, J*3)
        betas = body_pose_params.betas.cpu().numpy()         # (10,)
        transl = body_pose_params.transl.cpu().numpy()       # (T, 3)

        poses = np.concatenate([go, bp], axis=1)             # (T, 72) for SMPL
        betas_tiled = np.tile(betas[None], (T, 1))           # (T, 10)
        frames = [f"{i:06d}" for i in range(T)]

    out_path = os.path.join(output_dir, 'smpl_refined.npz')
    np.savez(
        out_path,
        poses=poses,
        betas=betas_tiled,
        transls=transl,
        gender=np.array(gender),
        model_type=np.array(model_type),
        frames=np.array(frames),
    )
    print(f"  [smpl] Saved refined params → {out_path}")


def _lookat(eye: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    """
    Build a (1, 4, 4) view matrix for an OpenCV camera (+Y down, +Z fwd).
    Camera Y axis (image-down) is aligned with world +Y (down).
    """
    import torch.nn.functional as F
    forward = F.normalize(target - eye, dim=0)
    world_down = torch.tensor([0., 1., 0.], device=device)
    if abs(float(torch.dot(forward, world_down))) > 0.99:
        world_down = torch.tensor([0., 0., 1.], device=device)
    right = F.normalize(torch.linalg.cross(world_down, forward), dim=0)
    down  = torch.linalg.cross(forward, right)
    R = torch.stack([right, down, forward])  # (3, 3)
    t = -(R @ eye)
    vm = torch.eye(4, device=device)
    vm[:3, :3] = R
    vm[:3, 3]  = t
    return vm.unsqueeze(0)  # (1, 4, 4)


def _orbit_viewmat_fixed(centroid: torch.Tensor, yaw_deg: float, device: str) -> torch.Tensor:
    """
    Build a view matrix for a camera orbiting ±yaw_deg (Y-axis rotation) around
    a fixed centroid. Original camera is at the origin looking along +Z.
    """
    angle = yaw_deg * np.pi / 180.0
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    Ry = torch.tensor([
        [ cos_a, 0., sin_a],
        [    0., 1.,    0.],
        [-sin_a, 0., cos_a],
    ], device=device, dtype=torch.float32)

    # Original camera is at origin; rotate (-centroid) around Y then re-centre
    cam_offset = -centroid                  # vector from centroid → origin
    eye = centroid + Ry @ cam_offset        # rotated camera position
    return _lookat(eye, centroid, device)


def save_renders_video(
    scene: GaussianScene,
    video_path: str,
    depth_folder: str,
    intrinsics,
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params,
    smpl_deformer,
    frame_indices,
    output_path: str,
    sh_degree: int = 3,
    device: str = 'cuda',
) -> None:
    """
    Render every frame and write a 4-panel video:
      original frame | rendered (original camera) | -30° orbit | +30° orbit

    Orbit cameras are fixed — centroid is computed once from the body position
    at frame 0, so the novel views don't drift as the person moves.
    """
    import cv2
    import subprocess
    from gsplat import rasterization

    from v2d.gsplat.lib.rasterizer import build_viewmat, build_K
    from v2d.gsplat.lib.optimization import compute_world_positions

    H, W = intrinsics.height, intrinsics.width
    viewmat_orig = build_viewmat(device)
    K_mat = build_K(intrinsics, device)

    frames_dir = os.path.join(os.path.dirname(output_path), '_render_tmp')
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _render_view(world_pos, viewmat):
        renders, _, _ = rasterization(
            means=world_pos,
            quats=scene.rotations,
            scales=scene.scales,
            opacities=scene.opacities,
            colors=scene.sh_features,
            viewmats=viewmat,
            Ks=K_mat,
            width=W,
            height=H,
            sh_degree=sh_degree,
            render_mode='RGB',
            backgrounds=torch.zeros(1, 3, device=device),
            packed=False,
        )
        return (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # Compute orbit centroid once from frame 0 body position
    with torch.no_grad():
        world_pos_f0 = compute_world_positions(
            scene, body_pose_params, obj_pose_params, smpl_deformer, frame_indices[0]
        )
        body_sel = scene.body_mask()
        if body_sel.any():
            orbit_centroid = world_pos_f0[body_sel].mean(dim=0)
        else:
            orbit_centroid = world_pos_f0.mean(dim=0)

    viewmat_m30 = _orbit_viewmat_fixed(orbit_centroid, -30., device)
    viewmat_p30 = _orbit_viewmat_fixed(orbit_centroid, +30., device)

    n_frames = len(frame_indices)
    label_h = 40
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, t in enumerate(frame_indices):
        print(f"  [render] frame {i+1}/{n_frames}", end='\r', flush=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok:
            break
        orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if orig.shape[0] != H or orig.shape[1] != W:
            orig = cv2.resize(orig, (W, H))

        with torch.no_grad():
            world_pos = compute_world_positions(
                scene, body_pose_params, obj_pose_params, smpl_deformer, t
            )
            rend_orig = _render_view(world_pos, viewmat_orig)
            rend_m30  = _render_view(world_pos, viewmat_m30)
            rend_p30  = _render_view(world_pos, viewmat_p30)

        # 4-panel with label bar
        canvas = np.zeros((H + label_h, 4 * W, 3), dtype=np.uint8)
        canvas[label_h:, 0*W:1*W] = orig
        canvas[label_h:, 1*W:2*W] = rend_orig
        canvas[label_h:, 2*W:3*W] = rend_m30
        canvas[label_h:, 3*W:4*W] = rend_p30
        cv2.putText(canvas, 'Original',   (10,         28), font, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, 'Rendered',   (W   + 10,   28), font, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, '-30 deg',    (2*W + 10,   28), font, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, '+30 deg',    (3*W + 10,   28), font, 0.9, (255, 255, 255), 2)

        cv2.imwrite(
            os.path.join(frames_dir, f"{i:06d}.png"),
            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
        )

    print()
    cap.release()

    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        output_path,
    ], check=True, capture_output=True)

    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"  [video] Saved 4-view video → {output_path}")


# ---------------------------------------------------------------------------
# Scene checkpoint  (save + reload without re-running optimisation)
# ---------------------------------------------------------------------------

def save_scene_checkpoint(
    scene: GaussianScene,
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params: Optional[ObjectPoseParams],
    output_dir: str,
) -> None:
    """
    Save all scene tensors + pose params to a single .pt file.

    Reload with load_scene_checkpoint() to reconstruct GaussianScene,
    BodyPoseParams and ObjectPoseParams without re-running optimisation.
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        # ---- GaussianScene raw parameters --------------------------------
        '_positions':           scene._positions.data.cpu(),
        '_rotations':           scene._rotations.data.cpu(),
        '_log_scales':          scene._log_scales.data.cpu(),
        '_opacities_raw':       scene._opacities_raw.data.cpu(),
        '_sh_dc':               scene._sh_dc.data.cpu(),
        '_sh_rest':             scene._sh_rest.data.cpu(),
        'entity_ids':           scene.entity_ids.cpu(),
        '_skinning_weights_raw': (scene._skinning_weights_raw.data.cpu()
                                  if scene._skinning_weights_raw is not None else None),
        'smpl_vertex_ids':      (scene.smpl_vertex_ids.cpu()
                                  if scene.smpl_vertex_ids is not None else None),
        # ---- Pose params -------------------------------------------------
        'body_pose_state':      (body_pose_params.state_dict()
                                  if body_pose_params is not None else None),
        'body_pose_T':          (body_pose_params.global_orient.shape[0]
                                  if body_pose_params is not None else None),
        'body_pose_J':          (body_pose_params.body_pose.shape[1] // 3
                                  if body_pose_params is not None else None),
        'obj_pose_state':       (obj_pose_params.state_dict()
                                  if obj_pose_params is not None else None),
        'obj_pose_T':           (obj_pose_params.rotations_6d.shape[0]
                                  if obj_pose_params is not None else None),
        'obj_pose_K':           (obj_pose_params.rotations_6d.shape[1]
                                  if obj_pose_params is not None else None),
    }
    path = os.path.join(output_dir, 'scene_checkpoint.pt')
    torch.save(ckpt, path)
    print(f"  [ckpt] Saved scene checkpoint ({scene.num_gaussians} Gaussians) → {path}")


def load_scene_checkpoint(
    path: str,
    device: str = 'cuda',
) -> tuple:
    """
    Load a scene checkpoint saved by save_scene_checkpoint().

    Returns (scene, body_pose_params, obj_pose_params) — same objects as
    produced by the optimisation, ready for rendering/inspection.
    """
    from v2d.gsplat.lib.deformation import BodyPoseParams, ObjectPoseParams

    ckpt = torch.load(path, map_location=device)

    # Reconstruct GaussianScene
    from v2d.gsplat.lib.scene import GaussianScene, sh_dc_to_rgb
    colors = sh_dc_to_rgb(ckpt['_sh_dc'][:, 0, :])  # DC band → RGB
    scene = GaussianScene(
        positions=ckpt['_positions'],
        colors=colors,
        entity_ids=ckpt['entity_ids'],
        skinning_weights=ckpt['_skinning_weights_raw'],
        smpl_vertex_ids=ckpt['smpl_vertex_ids'],
    ).to(device)
    scene._positions.data   = ckpt['_positions'].to(device)
    scene._rotations.data   = ckpt['_rotations'].to(device)
    scene._log_scales.data  = ckpt['_log_scales'].to(device)
    scene._opacities_raw.data = ckpt['_opacities_raw'].to(device)
    scene._sh_dc.data       = ckpt['_sh_dc'].to(device)
    scene._sh_rest.data     = ckpt['_sh_rest'].to(device)
    if scene._skinning_weights_raw is not None and ckpt['_skinning_weights_raw'] is not None:
        scene._skinning_weights_raw.data = ckpt['_skinning_weights_raw'].to(device)

    # Reconstruct BodyPoseParams
    body_pose_params = None
    if ckpt['body_pose_state'] is not None:
        body_pose_params = BodyPoseParams(
            ckpt['body_pose_T'], ckpt['body_pose_J'], device=device
        )
        body_pose_params.load_state_dict(ckpt['body_pose_state'])
        body_pose_params = body_pose_params.to(device)

    # Reconstruct ObjectPoseParams
    obj_pose_params = None
    if ckpt['obj_pose_state'] is not None:
        obj_pose_params = ObjectPoseParams(
            ckpt['obj_pose_T'], ckpt['obj_pose_K'], device=device
        )
        obj_pose_params.load_state_dict(ckpt['obj_pose_state'])
        obj_pose_params = obj_pose_params.to(device)

    return scene, body_pose_params, obj_pose_params


# ---------------------------------------------------------------------------
# Entity-coloured debug renders
# ---------------------------------------------------------------------------

# Flat colours per entity type (RGB in [0,1])
_ENTITY_PALETTE = {
    ENTITY_BACKGROUND: [0.75, 0.75, 0.75],   # light gray
    ENTITY_BODY:       [0.20, 0.45, 0.90],   # blue
}
_OBJECT_COLOURS = [
    [0.90, 0.20, 0.20],  # red
    [0.15, 0.75, 0.30],  # green
    [0.90, 0.75, 0.10],  # yellow
    [0.80, 0.30, 0.85],  # purple
]


def _build_entity_color_tensor(scene: GaussianScene, device: str) -> torch.Tensor:
    """(N, 3) flat colour for each Gaussian based on its entity_id."""
    colors = torch.zeros(scene.num_gaussians, 3, device=device)
    for eid, rgb in _ENTITY_PALETTE.items():
        mask = scene.entity_ids == eid
        colors[mask] = torch.tensor(rgb, device=device)
    for k in range(scene.n_objects()):
        mask = scene.object_mask(k)
        rgb = _OBJECT_COLOURS[k % len(_OBJECT_COLOURS)]
        colors[mask] = torch.tensor(rgb, device=device)
    return colors  # (N, 3)


def save_entity_renders(
    scene: GaussianScene,
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params,
    smpl_deformer,
    intrinsics,
    frame_indices,
    output_dir: str,
    device: str = 'cuda',
) -> None:
    """
    Save per-frame debug renders:
      debug/entity_colors.mp4          — Gaussians flat-coloured by entity
      debug/masks/entity_{id}/{i:06d}.png — alpha mask per entity per frame
    """
    import cv2, subprocess, shutil
    from gsplat import rasterization
    from v2d.gsplat.lib.rasterizer import build_viewmat, build_K
    from v2d.gsplat.lib.optimization import compute_world_positions

    os.makedirs(output_dir, exist_ok=True)
    H, W = intrinsics.height, intrinsics.width
    viewmat = build_viewmat(device)
    K_mat   = build_K(intrinsics, device)

    entity_colors = _build_entity_color_tensor(scene, device)   # (N, 3)

    # Unique entity ids present in scene
    unique_eids = scene.entity_ids.unique().tolist()

    # Temp dirs
    color_frames_dir = os.path.join(output_dir, '_ec_tmp')
    os.makedirs(color_frames_dir, exist_ok=True)
    mask_dirs = {}
    for eid in unique_eids:
        d = os.path.join(output_dir, 'masks', f'entity_{int(eid)}')
        os.makedirs(d, exist_ok=True)
        mask_dirs[int(eid)] = d

    fps = 30.0
    n_frames = len(frame_indices)

    for i, t in enumerate(frame_indices):
        print(f"  [entity render] frame {i+1}/{n_frames}", end='\r', flush=True)
        with torch.no_grad():
            world_pos = compute_world_positions(
                scene, body_pose_params, obj_pose_params, smpl_deformer, t
            )

            # ---- Entity-coloured render (flat colours, SH degree 0) ----
            renders, alphas, _ = rasterization(
                means=world_pos,
                quats=scene.rotations,
                scales=scene.scales,
                opacities=scene.opacities,
                colors=entity_colors.unsqueeze(1),   # (N, 1, 3)
                viewmats=viewmat,
                Ks=K_mat,
                width=W,
                height=H,
                sh_degree=0,
                render_mode='RGB',
                backgrounds=torch.tensor([[0.0, 0.0, 0.0]], device=device),
                packed=False,
            )
            rgb_np = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(color_frames_dir, f'{i:06d}.png'),
                cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            )

            # ---- Per-entity alpha masks ---------------------------------
            for eid in unique_eids:
                mask_bool = scene.entity_ids == eid
                ops_masked = scene.opacities.clone()
                ops_masked[~mask_bool] = 0.0
                _, mask_alpha, _ = rasterization(
                    means=world_pos,
                    quats=scene.rotations,
                    scales=scene.scales,
                    opacities=ops_masked,
                    colors=entity_colors.unsqueeze(1),
                    viewmats=viewmat,
                    Ks=K_mat,
                    width=W,
                    height=H,
                    sh_degree=0,
                    render_mode='RGB',
                    packed=False,
                )
                alpha_np = (mask_alpha[0, :, :, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(mask_dirs[int(eid)], f'{i:06d}.png'), alpha_np)

    # Encode entity-colour video
    video_path = os.path.join(output_dir, 'entity_colors.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(color_frames_dir, '%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        video_path,
    ], check=True, capture_output=True)
    print()  # newline after \r progress
    shutil.rmtree(color_frames_dir, ignore_errors=True)
    print(f"  [debug] Entity-colour video → {video_path}")
    print(f"  [debug] Entity masks → {os.path.join(output_dir, 'masks')}/")


# ---------------------------------------------------------------------------
# Novel-viewpoint orbit renders
# ---------------------------------------------------------------------------

def save_orbit_renders(
    scene: GaussianScene,
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params,
    smpl_deformer,
    intrinsics,
    frame_t: int,
    output_dir: str,
    device: str = 'cuda',
) -> None:
    """
    Render front and side views centered on the body entity at a single frame.
    Saves view_front.png, view_side.png, and a side-by-side views.png.

    Coordinate system: depth-space, OpenCV convention (+Y down, +Z forward).
    Front view: camera behind person (−Z), looking +Z.
    Side view:  camera to person's right (+X), looking −X.
    """
    import cv2
    from gsplat import rasterization
    from v2d.gsplat.lib.rasterizer import build_K
    from v2d.gsplat.lib.optimization import compute_world_positions

    os.makedirs(output_dir, exist_ok=True)
    H, W = intrinsics.height, intrinsics.width
    K_mat = build_K(intrinsics, device)

    with torch.no_grad():
        world_pos = compute_world_positions(
            scene, body_pose_params, obj_pose_params, smpl_deformer, frame_t
        )

        # Centre on body Gaussians; fall back to whole scene if no body present
        body_sel = scene.body_mask()
        if body_sel.any():
            body_pos = world_pos[body_sel]
            centroid = body_pos.mean(dim=0)
            body_extent = float((body_pos - centroid).norm(dim=-1).max())
            radius = max(body_extent * 3.0, 1.5)
        else:
            centroid = world_pos.mean(dim=0)
            radius = float((world_pos - centroid).norm(dim=-1).mean()) * 2.0
            radius = max(radius, 1.5)

    def _render(viewmat: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            renders, _, _ = rasterization(
                means=world_pos,
                quats=scene.rotations,
                scales=scene.scales,
                opacities=scene.opacities,
                colors=scene.sh_features,
                viewmats=viewmat,
                Ks=K_mat,
                width=W,
                height=H,
                sh_degree=3,
                render_mode='RGB',
                backgrounds=torch.zeros(1, 3, device=device),
                packed=False,
            )
        return (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # Front: camera at centroid − radius·Z, looking +Z
    eye_front = centroid.clone()
    eye_front[2] = centroid[2] - radius
    print("  [views] Rendering front view...", flush=True)
    rgb_front = _render(_lookat(eye_front, centroid, device))

    # Side: camera at centroid + radius·X, looking −X
    eye_side = centroid.clone()
    eye_side[0] = centroid[0] + radius
    print("  [views] Rendering side view...", flush=True)
    rgb_side = _render(_lookat(eye_side, centroid, device))

    # Save individual PNGs
    cv2.imwrite(os.path.join(output_dir, 'view_front.png'),
                cv2.cvtColor(rgb_front, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'view_side.png'),
                cv2.cvtColor(rgb_side,  cv2.COLOR_RGB2BGR))

    # Side-by-side with labels
    label_h = 60
    canvas = np.zeros((H + label_h, 2 * W, 3), dtype=np.uint8)
    canvas[label_h:, :W]  = rgb_front
    canvas[label_h:, W:]  = rgb_side
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Front', (20, 42),   font, 1.4, (255, 255, 255), 2)
    cv2.putText(canvas, 'Side',  (W + 20, 42), font, 1.4, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, 'views.png'),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    print(f"  [views] Saved front + side views → {output_dir}/views.png")
