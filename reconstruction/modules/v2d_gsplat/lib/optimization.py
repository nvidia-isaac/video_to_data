"""
Alternating canonical / pose optimization for 4D Gaussian Splatting.

Each cycle has two sub-phases:
  Canonical — freeze body/object poses, optimize Gaussian geometry + densify.
  Pose      — freeze Gaussian geometry, optimize body LBS pose + object SE(3).

A final joint refinement phase runs at 0.1× LR after all cycles completes.
"""

import os
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.gsplat.lib.scene import GaussianScene, ENTITY_BODY, ENTITY_OBJECT_BASE
from v2d.gsplat.lib.deformation import SmplDeformer, BodyPoseParams, ObjectPoseParams, ExposureParams, apply_lbs
from v2d.gsplat.lib.rasterizer import render, render_entity_silhouette, build_viewmat, build_K
from v2d.gsplat.lib.losses import compute_total_loss, loss_mask, loss_mask_asymmetric, loss_anchor
from v2d.gsplat.lib.densification import densify_and_prune


@dataclass
class OptimConfig:
    # ---- Optimization mode --------------------------------------------------
    # alternating=True  (default): each cycle has a canonical sub-phase
    #   (Gaussians + densification, poses frozen) then a pose sub-phase
    #   (poses only, Gaussians frozen).
    # alternating=False (joint):   each cycle optimises all parameters together
    #   for iterations_canonical_per_cycle + iterations_pose_per_cycle iters
    #   with densification throughout.
    alternating: bool = True
    n_cycles: int = 3
    iterations_canonical_per_cycle: int = 1000   # iters for canonical sub-phase (or joint)
    iterations_pose_per_cycle: int = 500          # iters for pose sub-phase (alternating only)
    # ---- Final joint refinement ---------------------------------------------
    iterations_refine: int = 1000                 # joint low-LR polish after cycles
    # ---- Final pose sweep ---------------------------------------------------
    # After refinement, canonical is frozen and every frame in the full video
    # gets a per-frame pose update (catches frames skipped by frame_step).
    n_pose_sweep_passes: int = 1                  # passes over all frames
    # ---- Learning rates -----------------------------------------------------
    # lr_scale multiplies every per-group LR below (quick global tuning knob).
    lr_scale: float = 1.0
    lr_positions: float = 1.6e-4
    lr_rotations: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 1e-3
    lr_sh_rest: float = 1e-4
    lr_skinning: float = 1e-4
    lr_body_pose: float = 1e-3    # global_orient (root rotation)
    lr_body_joints: float = 0.0   # body_pose (joint angles) — 0 = lock to NLF
    lr_body_shape: float = 1e-4
    lr_body_transl: float = 1e-3
    lr_obj_pose: float = 1e-3
    lr_exposure: float = 1e-2   # per-frame log-exposure learning rate
    # L2 penalty on log_exposure — keeps values near 0 (neutral), prevents
    # exposure from absorbing real scene colour variation.
    weight_exposure_reg: float = 0.1
    # ---- Densification (canonical sub-phase only) ---------------------------
    densify_every: int = 100
    grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.005
    max_gaussians: int = 500_000
    # Prune any Gaussian whose max scale exceeds (max_scale_factor * scene_extent).
    # Removes needle/streak artifacts from over-grown Gaussians. Applies to all
    # entities (background, body, object). 0.0 = disabled.
    max_scale_factor: float = 0.1
    # Reset all Gaussian opacities to a small value every N canonical/joint iters.
    # Forces Gaussians to re-earn their opacity; culls dead elongated floaters.
    # 0 = disabled. Typical: 500–1000.
    reset_opacity_every: int = 500
    # ---- Frames / batching --------------------------------------------------
    batch_size: int = 4
    # ---- Loss weights -------------------------------------------------------
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'rgb': 1.0, 'ssim': 0.2, 'depth': 0.1, 'mask': 0.0,
        'smooth': 0.01, 'skinning': 0.01, 'entity_mask': 1.0,
    })
    sh_degree: int = 3
    train_scale: float = 0.5
    entity_mask_interval: int = 5
    weight_obj_pose_smooth: float = 0.0
    weight_body_pose_smooth: float = 0.0
    weight_body_anchor: float = 0.0
    weight_obj_anchor: float = 0.0
    # Weight applied to body entity-mask loss outside the SAM2 body silhouette.
    # 0.1 = gentle (default, avoids penalising occluded parts).
    # 0.5–1.0 = aggressive ghost elimination at hands/feet (faster convergence).
    body_mask_outside_weight: float = 0.5
    # Penalise anisotropic (needle/flat) Gaussians by penalising the spread
    # between max and min log-scale across the 3 axes.
    # 0.0 = disabled; 0.01–0.1 = gentle; higher = more spherical Gaussians.
    # Improves novel-view (side-view) consistency at the cost of some detail.
    weight_isotropy: float = 0.0
    device: str = 'cuda'


def _load_frame_rgb(video_path: str, frame_idx: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _lookat(eye: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    """Build a (1, 4, 4) OpenCV-convention view matrix (+Y down, +Z forward)."""
    z = torch.nn.functional.normalize(target - eye, dim=0)
    up = torch.tensor([0., -1., 0.], device=device)
    x = torch.nn.functional.normalize(torch.linalg.cross(z, up), dim=0)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], dim=0)        # (3, 3) rows = cam axes
    t = -(R @ eye)                            # (3,)
    vm = torch.eye(4, device=device)
    vm[:3, :3] = R
    vm[:3,  3] = t
    return vm.unsqueeze(0)                    # (1, 4, 4)


def _orbit_viewmat(centroid: torch.Tensor, yaw_deg: float, device: str) -> torch.Tensor:
    """Camera orbiting ±yaw_deg (Y-axis) around centroid; original camera at origin, looking +Z."""
    yaw = torch.tensor(yaw_deg * np.pi / 180., dtype=torch.float32, device=device)
    c, s = torch.cos(yaw), torch.sin(yaw)
    Ry = torch.stack([
        torch.stack([ c, torch.zeros(1, device=device).squeeze(), s]),
        torch.tensor([0., 1., 0.], device=device),
        torch.stack([-s, torch.zeros(1, device=device).squeeze(), c]),
    ])
    cam_offset = -centroid
    eye = centroid + Ry @ cam_offset
    return _lookat(eye, centroid, device)


def _load_depth_tensor(depth_folder: str, frame_idx: int, device: str) -> torch.Tensor:
    path = os.path.join(depth_folder, f"{frame_idx:06d}.png")
    if not os.path.exists(path):
        return None
    depth_np = DepthImage.load(path).depth  # (H, W) float32 metres
    return torch.tensor(depth_np, dtype=torch.float32, device=device)


def _load_mask_tensor(masks_dir: str, object_id: int, frame_idx: int, device: str) -> Optional[torch.Tensor]:
    from PIL import Image
    path = os.path.join(masks_dir, str(object_id), f"{frame_idx:06d}.png")
    if not os.path.exists(path):
        return None
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return torch.tensor(mask > 127, dtype=torch.float32, device=device)


def compute_world_positions(
    scene: GaussianScene,
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params: Optional[ObjectPoseParams],
    smpl_deformer: Optional[SmplDeformer],
    frame_t: int,
) -> torch.Tensor:
    """
    Return (N, 3) world-space positions for all Gaussians at frame t.

    Background: identity (use canonical positions).
    Body: LBS via SMPL using current body_pose_params.
    Objects: SE(3) rigid transform using obj_pose_params.
    """
    world_pos = scene.positions.clone()

    # ---- Body -------------------------------------------------------
    body_mask = scene.body_mask()
    if body_mask.any() and body_pose_params is not None and smpl_deformer is not None:
        go, bp, betas, transl = body_pose_params.frame(frame_t)

        if scene.skinning_weights is not None:
            # Full LBS with learned skinning weights
            A = smpl_deformer.get_joint_transforms(go, bp, betas, transl)  # (1, J, 4, 4)
            A = A.squeeze(0)  # (J, 4, 4)
            canonical_body = scene.positions[body_mask]  # (N_body, 3)
            sw = scene.skinning_weights  # (N_body, J) softmax
            world_body = apply_lbs(canonical_body, sw, A, transl.squeeze(0))
        else:
            # Simpler: use SMPL vertex positions directly
            verts = smpl_deformer.get_posed_vertices(go, bp, betas, transl)  # (1, V, 3)
            verts = verts.squeeze(0)  # (V, 3)
            vertex_ids = scene.smpl_vertex_ids  # (N_body,)
            world_body = verts[vertex_ids]

        world_pos = world_pos.clone()
        world_pos[body_mask] = world_body

    # ---- Objects ----------------------------------------------------
    if obj_pose_params is not None:
        for rid in range(scene.n_objects()):
            obj_mask = scene.object_mask(rid)
            if not obj_mask.any():
                continue
            R, t = obj_pose_params.get_transform(frame_t, rid)
            canonical_obj = scene.positions[obj_mask]
            world_obj = canonical_obj @ R.T + t.unsqueeze(0)
            world_pos = world_pos.clone()
            world_pos[obj_mask] = world_obj

    return world_pos


def _setup_optimizer(
    scene: GaussianScene,
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params: Optional[ObjectPoseParams],
    cfg: OptimConfig,
    mode: str = 'canonical',
    exposure_params: Optional['ExposureParams'] = None,
) -> optim.Adam:
    """
    Build Adam optimizer.
      mode='canonical' — Gaussian geometry active, pose params frozen.
      mode='pose'      — Pose params active, Gaussian geometry frozen.
      mode='joint'     — All params active at full LR (used when alternating=False).
      mode='refine'    — All params at 0.1× LR for final polish.
    """
    s = cfg.lr_scale
    param_groups = [
        {'params': [scene._positions],      'lr': cfg.lr_positions * s,  'name': 'positions'},
        {'params': [scene._rotations],      'lr': cfg.lr_rotations * s,  'name': 'rotations'},
        {'params': [scene._log_scales],     'lr': cfg.lr_scales    * s,  'name': 'scales'},
        {'params': [scene._opacities_raw],  'lr': cfg.lr_opacities * s,  'name': 'opacities'},
        {'params': [scene._sh_dc],          'lr': cfg.lr_sh_dc     * s,  'name': 'sh_dc'},
        {'params': [scene._sh_rest],        'lr': cfg.lr_sh_rest   * s,  'name': 'sh_rest'},
    ]

    if scene._skinning_weights_raw is not None:
        param_groups.append({
            'params': [scene._skinning_weights_raw],
            'lr': cfg.lr_skinning * s,
            'name': 'skinning',
        })

    if body_pose_params is not None:
        param_groups += [
            {'params': [body_pose_params.global_orient], 'lr': cfg.lr_body_pose   * s, 'name': 'go'},
            {'params': [body_pose_params.body_pose],     'lr': cfg.lr_body_joints * s, 'name': 'bp'},
            {'params': [body_pose_params.betas],         'lr': cfg.lr_body_shape  * s, 'name': 'betas'},
            {'params': [body_pose_params.transl],        'lr': cfg.lr_body_transl * s, 'name': 'transl'},
        ]

    if obj_pose_params is not None:
        param_groups += [
            {'params': [obj_pose_params.rotations_6d],  'lr': cfg.lr_obj_pose * s, 'name': 'obj_rot'},
            {'params': [obj_pose_params.translations],  'lr': cfg.lr_obj_pose * s, 'name': 'obj_t'},
        ]

    # Exposure is always active in every mode — it's a per-frame appearance
    # correction, not a geometry or pose param.
    if exposure_params is not None:
        param_groups.append({
            'params': [exposure_params.log_exposure],
            'lr': cfg.lr_exposure,
            'name': 'exposure',
        })

    _gaussian_geom = {'positions', 'rotations', 'scales', 'sh_dc', 'sh_rest', 'skinning'}
    _pose_params   = {'go', 'bp', 'betas', 'transl', 'obj_rot', 'obj_t'}

    if mode == 'canonical':
        for pg in param_groups:
            if pg['name'] in _pose_params:
                pg['lr'] = 0.0
    elif mode == 'pose':
        for pg in param_groups:
            if pg['name'] in _gaussian_geom:
                pg['lr'] = 0.0
    elif mode == 'refine':
        for pg in param_groups:
            pg['lr'] *= 0.1
    # mode='joint': all LRs left at full value

    return optim.Adam(param_groups, lr=0.0, eps=1e-15)


def _compute_batch_loss(
    scene: GaussianScene,
    batch_frames: List[int],
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params: Optional[ObjectPoseParams],
    smpl_deformer: Optional[SmplDeformer],
    cfg: OptimConfig,
    viewmat, K_tr, H_tr: int, W_tr: int,
    _cache_rgb: Dict,
    _cache_depth: Dict,
    _cache_masks: Dict,
    human_oids: List[int],
    object_oids: List[int],
    device: str,
    compute_entity_mask: bool,
    exposure_params: Optional[ExposureParams] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Forward pass over batch_frames; return (total_loss, per-term breakdown)."""
    batch_loss = torch.tensor(0.0, device=device)
    loss_terms_accum: Dict[str, float] = {}

    for t in batch_frames:
        world_pos = compute_world_positions(
            scene, body_pose_params, obj_pose_params, smpl_deformer, t
        )
        result = render(scene, world_pos, viewmat, K_tr, H_tr, W_tr, sh_degree=cfg.sh_degree)

        # Apply per-frame exposure correction before RGB loss so Gaussians learn
        # appearance at neutral exposure and camera auto-exposure is absorbed here.
        if exposure_params is not None:
            rendered_rgb = result.rgb * exposure_params.get(t)
        else:
            rendered_rgb = result.rgb

        target_rgb   = _cache_rgb[t]
        target_depth = _cache_depth[t]
        target_body_mask = _cache_masks.get((human_oids[0], t)) if human_oids else None

        combined_mask = target_body_mask
        if combined_mask is None and object_oids:
            for oid in object_oids:
                om = _cache_masks.get((oid, t))
                if om is not None:
                    combined_mask = om if combined_mask is None else (combined_mask + om).clamp(0, 1)

        total, terms = compute_total_loss(
            rendered_rgb=rendered_rgb,
            target_rgb=target_rgb,
            rendered_depth=result.depth,
            target_depth=target_depth,
            rendered_alpha=result.alpha,
            target_mask=combined_mask,
            body_pose_params=body_pose_params,
            scene=scene,
            weights=cfg.loss_weights,
        )

        if compute_entity_mask:
            entity_mask_weight = cfg.loss_weights.get('entity_mask', 1.0)
            entity_mask_loss = torch.tensor(0.0, device=device)
            n_entity_losses = 0
            if human_oids:
                hm = _cache_masks.get((human_oids[0], t))
                if hm is not None:
                    body_sel = scene.body_mask()
                    if body_sel.any():
                        body_alpha = render_entity_silhouette(
                            scene, body_sel, world_pos, viewmat, K_tr, H_tr, W_tr
                        )
                        entity_mask_loss = entity_mask_loss + loss_mask_asymmetric(
                            body_alpha, hm, outside_weight=cfg.body_mask_outside_weight
                        )
                        n_entity_losses += 1
            for rid, oid in enumerate(object_oids):
                obj_sam2_mask = _cache_masks.get((oid, t))
                if obj_sam2_mask is None:
                    continue
                entity_sel = scene.entity_ids == (ENTITY_OBJECT_BASE + rid)
                if entity_sel.any():
                    entity_alpha = render_entity_silhouette(
                        scene, entity_sel, world_pos, viewmat, K_tr, H_tr, W_tr
                    )
                    entity_mask_loss = entity_mask_loss + loss_mask_asymmetric(entity_alpha, obj_sam2_mask)
                    n_entity_losses += 1
            if n_entity_losses > 0:
                entity_mask_loss = entity_mask_loss / n_entity_losses
                total = total + entity_mask_weight * entity_mask_loss
                terms['entity_mask'] = entity_mask_loss.item()
            else:
                terms['entity_mask'] = 0.0
        else:
            terms['entity_mask'] = 0.0

        batch_loss = batch_loss + total / len(batch_frames)
        for k, v in terms.items():
            loss_terms_accum[k] = loss_terms_accum.get(k, 0.0) + v / len(batch_frames)

    # Anchor losses (canonical-space, frame-independent)
    if cfg.weight_body_anchor > 0.0 and smpl_deformer is not None:
        body_mask = scene.body_mask()
        if body_mask.any():
            with torch.no_grad():
                v_rest = smpl_deformer.get_rest_vertices(body_pose_params.betas)
            target_body = v_rest[scene.smpl_vertex_ids].detach()
            batch_loss = batch_loss + cfg.weight_body_anchor * loss_anchor(
                scene._positions[body_mask], target_body
            )

    if cfg.weight_obj_anchor > 0.0:
        for rid in range(scene.n_objects()):
            obj_mask_a = scene.object_mask(rid)
            initial = scene._initial_obj_positions.get(rid)
            if obj_mask_a.any() and initial is not None:
                batch_loss = batch_loss + cfg.weight_obj_anchor * loss_anchor(
                    scene._positions[obj_mask_a], initial.to(device)
                )

    # Object pose smoothness
    if obj_pose_params is not None and cfg.weight_obj_pose_smooth > 0.0:
        dt = obj_pose_params.translations[1:] - obj_pose_params.translations[:-1]
        dr = obj_pose_params.rotations_6d[1:] - obj_pose_params.rotations_6d[:-1]
        pose_smooth_loss = (dt ** 2).mean() + (dr ** 2).mean()
        batch_loss = batch_loss + cfg.weight_obj_pose_smooth * pose_smooth_loss

    # Body pose smoothness
    if body_pose_params is not None and cfg.weight_body_pose_smooth > 0.0:
        dgo = body_pose_params.global_orient[1:] - body_pose_params.global_orient[:-1]
        dbp = body_pose_params.body_pose[1:]    - body_pose_params.body_pose[:-1]
        dtr = body_pose_params.transl[1:]       - body_pose_params.transl[:-1]
        body_smooth_loss = (dgo ** 2).mean() + (dbp ** 2).mean() + (dtr ** 2).mean()
        batch_loss = batch_loss + cfg.weight_body_pose_smooth * body_smooth_loss

    # Isotropy regularisation — penalise anisotropic (needle/flat) Gaussians.
    # max_log_scale - min_log_scale is the log ratio between the longest and
    # shortest axis; minimising it pushes Gaussians toward spherical shapes,
    # which improves appearance from novel (side) viewpoints.
    if cfg.weight_isotropy > 0.0:
        anisotropy = (
            scene._log_scales.max(dim=-1).values - scene._log_scales.min(dim=-1).values
        ).mean()
        batch_loss = batch_loss + cfg.weight_isotropy * anisotropy
        loss_terms_accum['isotropy'] = anisotropy.item()

    # Exposure regularisation — L2 on log_exposure keeps values near 0 (neutral).
    if exposure_params is not None and cfg.weight_exposure_reg > 0.0:
        exposure_reg = exposure_params.log_exposure.pow(2).mean()
        batch_loss = batch_loss + cfg.weight_exposure_reg * exposure_reg
        loss_terms_accum['exposure_reg'] = exposure_reg.item()

    return batch_loss, loss_terms_accum


def run_optimization(
    scene: GaussianScene,
    video_path: str,
    depth_folder: str,
    intrinsics: CameraIntrinsics,
    masks_dir: str,
    entity_role_map: Dict[int, str],   # {object_id: role}
    frame_indices: List[int],          # training frames (downsampled by frame_step)
    all_frame_indices: List[int],      # every frame in the video (for pose sweep)
    body_pose_params: Optional[BodyPoseParams],
    obj_pose_params: Optional[ObjectPoseParams],
    smpl_deformer: Optional[SmplDeformer],
    cfg: OptimConfig,
    output_dir: str,
    total_frames: int = 0,
) -> GaussianScene:
    """
    Run alternating canonical/pose cycles then a final refinement.
    Checkpoints renders every 500 iterations to output_dir/renders/.
    """
    device = cfg.device
    renders_dir = os.path.join(output_dir, 'renders')
    os.makedirs(renders_dir, exist_ok=True)

    H, W = intrinsics.height, intrinsics.width
    viewmat = build_viewmat(device)
    K = build_K(intrinsics, device)

    # Downscaled training resolution — cuts render cost quadratically with no
    # meaningful quality loss (3DGS optimises world-space params, not pixels).
    s = cfg.train_scale
    if s != 1.0:
        H_tr = max(1, int(H * s))
        W_tr = max(1, int(W * s))
        K_tr = K.clone()
        K_tr[0, 0, 0] *= s   # fx
        K_tr[0, 1, 1] *= s   # fy
        K_tr[0, 0, 2] *= s   # cx
        K_tr[0, 1, 2] *= s   # cy
        print(f"  [optim] Training at {W_tr}×{H_tr} (scale={s}); final renders at {W}×{H}")
    else:
        H_tr, W_tr, K_tr = H, W, K

    # Object IDs by role for mask lookup
    human_oids = [oid for oid, r in entity_role_map.items() if r == 'human']
    object_oids = sorted([oid for oid, r in entity_role_map.items() if r == 'object'])

    # ------------------------------------------------------------------ #
    # Preload all training data into GPU memory (eliminates per-iter disk I/O)
    # ------------------------------------------------------------------ #
    all_oids = human_oids + object_oids
    print(f"  [optim] Preloading {len(frame_indices)} frames × "
          f"{1 + len(all_oids)} targets into GPU memory...", flush=True)
    _cache_rgb:   Dict[int, torch.Tensor] = {}
    _cache_depth: Dict[int, Optional[torch.Tensor]] = {}
    _cache_masks: Dict[tuple, Optional[torch.Tensor]] = {}  # (oid, t) → mask

    import torch.nn.functional as F_nn

    def _resize_to_tr(t2d: torch.Tensor) -> torch.Tensor:
        """Bilinear-downsample a (H0, W0) mask/depth to (H_tr, W_tr)."""
        if s == 1.0 or (t2d.shape[0] == H_tr and t2d.shape[1] == W_tr):
            return t2d
        return F_nn.interpolate(
            t2d.unsqueeze(0).unsqueeze(0).float(), size=(H_tr, W_tr),
            mode='bilinear', align_corners=False,
        ).squeeze()

    # Pre-build morphological fill function for body masks.
    # Body segmentation often has gaps (dark clothing breaks the silhouette).
    # Without filling, the scene mask loss pushes body opacity DOWN at those
    # pixels every iteration — RGB loss can't compensate once colour converges.
    from scipy.ndimage import binary_closing, binary_fill_holes

    def _fill_body_mask(m: torch.Tensor) -> torch.Tensor:
        """Close intra-mask gaps and fill holes; returns float32 tensor."""
        arr = m.cpu().numpy() > 0.5
        arr = binary_fill_holes(binary_closing(arr, iterations=12))
        return torch.tensor(arr.astype(np.float32), device=m.device)

    for t in frame_indices:
        _cache_rgb[t]   = _load_target_rgb(video_path, t, H_tr, W_tr, device)
        raw_d = _load_depth_tensor(depth_folder, t, device)
        _cache_depth[t] = _resize_to_tr(raw_d) if raw_d is not None else None
        for oid in all_oids:
            raw_m = _load_mask_tensor(masks_dir, oid, t, device)
            if raw_m is not None and oid in human_oids:
                raw_m = _fill_body_mask(raw_m)
            _cache_masks[(oid, t)] = _resize_to_tr(raw_m) if raw_m is not None else None

    print(f"  [optim] Data cache ready.", flush=True)

    # Per-frame exposure correction (absorbs camera auto-exposure variation).
    n_frames_total = total_frames if total_frames > 0 else (max(all_frame_indices) + 1)
    exposure_params = ExposureParams(n_frames_total, device=device) if cfg.lr_exposure > 0.0 else None
    if exposure_params is not None:
        print(f"  [optim] Per-frame exposure learning enabled (lr={cfg.lr_exposure})")

    # ------------------------------------------------------------------ #
    # Shared keyword args for _compute_batch_loss (everything except
    # scene, batch_frames, compute_entity_mask — those vary per call).
    # ------------------------------------------------------------------ #
    _loss_kwargs = dict(
        body_pose_params=body_pose_params,
        obj_pose_params=obj_pose_params,
        smpl_deformer=smpl_deformer,
        cfg=cfg,
        viewmat=viewmat,
        K_tr=K_tr,
        H_tr=H_tr,
        W_tr=W_tr,
        _cache_rgb=_cache_rgb,
        _cache_depth=_cache_depth,
        _cache_masks=_cache_masks,
        human_oids=human_oids,
        object_oids=object_oids,
        device=device,
        exposure_params=exposure_params,
    )

    def _sample_batch() -> List[int]:
        if cfg.batch_size >= len(frame_indices):
            return list(frame_indices)
        return random.sample(frame_indices, cfg.batch_size)

    def _log(label: str, i: int, n_iters: int, loss_val: float, terms: Dict, n_gauss: int):
        terms_str = ' '.join(f"{k}={v:.4f}" for k, v in terms.items())
        line = f"  {label}[{i+1}/{n_iters}] loss={loss_val:.4f}  {terms_str}  N={n_gauss}"
        if (i + 1) % 50 == 0 or i == 0 or i == n_iters - 1:
            print(line)
        else:
            print(line, end='\r', flush=True)

    # Orbit cameras fixed at frame-0 body centroid (same as final render video).
    with torch.no_grad():
        _wp_f0 = compute_world_positions(
            scene, body_pose_params, obj_pose_params, smpl_deformer, frame_indices[0]
        )
        _body_sel = scene.body_mask()
        _orbit_centroid = _wp_f0[_body_sel].mean(dim=0) if _body_sel.any() else _wp_f0.mean(dim=0)
    _viewmat_m30 = _orbit_viewmat(_orbit_centroid, -30., device)
    _viewmat_p30 = _orbit_viewmat(_orbit_centroid, +30., device)

    def _checkpoint(tag: str, frame_t: int):
        try:
            orig_rgb = _load_frame_rgb(video_path, frame_t)  # (H, W, 3) uint8
            if orig_rgb.shape[0] != H or orig_rgb.shape[1] != W:
                orig_rgb = cv2.resize(orig_rgb, (W, H))
        except Exception:
            orig_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        with torch.no_grad():
            _wp = compute_world_positions(
                scene, body_pose_params, obj_pose_params, smpl_deformer, frame_t
            )
            rend_orig = _render_np(scene, _wp, viewmat,        K, H, W, cfg.sh_degree, device)
            rend_m30  = _render_np(scene, _wp, _viewmat_m30,   K, H, W, cfg.sh_degree, device)
            rend_p30  = _render_np(scene, _wp, _viewmat_p30,   K, H, W, cfg.sh_degree, device)

        label_h = 40
        canvas = np.zeros((H + label_h, 4 * W, 3), dtype=np.uint8)
        canvas[label_h:, 0*W:1*W] = orig_rgb
        canvas[label_h:, 1*W:2*W] = rend_orig
        canvas[label_h:, 2*W:3*W] = rend_m30
        canvas[label_h:, 3*W:4*W] = rend_p30
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, 'Original', (10,       28), font, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, 'Rendered', (W   + 10, 28), font, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, '-30 deg',  (2*W + 10, 28), font, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, '+30 deg',  (3*W + 10, 28), font, 0.9, (255, 255, 255), 2)

        out_path = os.path.join(renders_dir, f"{tag}.png")
        try:
            cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"  [render save failed: {e}]")

    total_iter = 0

    # ------------------------------------------------------------------ #
    # N cycles — alternating (canonical → pose) or joint (all params)
    # ------------------------------------------------------------------ #
    def _run_cycle_phase(opt_mode, n_iters, log_label, do_densify):
        """
        Run one optimization phase. Returns (scene, optimizer) since densification
        may replace the scene object.
        """
        nonlocal scene
        optimizer = _setup_optimizer(scene, body_pose_params, obj_pose_params, cfg, opt_mode, exposure_params)
        pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)
        pos_grad_count = torch.zeros(scene.num_gaussians, device=device)

        for i in range(n_iters):
            batch_frames = _sample_batch()
            compute_entity_mask = (
                cfg.loss_weights.get('entity_mask', 1.0) > 0
                and (i % cfg.entity_mask_interval == 0)
            )
            optimizer.zero_grad()
            batch_loss, terms = _compute_batch_loss(
                scene, batch_frames, compute_entity_mask=compute_entity_mask,
                **_loss_kwargs
            )
            batch_loss.backward()

            if do_densify and scene._positions.grad is not None:
                pos_grad_accum += scene._positions.grad.norm(dim=-1).detach()
                pos_grad_count += 1

            optimizer.step()
            nonlocal total_iter
            total_iter += 1
            _log(f"[{log_label}]", i, n_iters, batch_loss.item(), terms, scene.num_gaussians)

            if do_densify and (i + 1) % cfg.densify_every == 0:
                grad_avg = pos_grad_accum / pos_grad_count.clamp(min=1)
                scene_extent = float(scene.positions.detach().norm(dim=-1).max())
                scene = densify_and_prune(
                    scene, grad_avg,
                    grad_threshold=cfg.grad_threshold,
                    prune_opacity_threshold=cfg.prune_opacity_threshold,
                    max_scene_extent=scene_extent,
                    max_gaussians=cfg.max_gaussians,
                    max_scale_factor=cfg.max_scale_factor,
                )
                optimizer = _setup_optimizer(scene, body_pose_params, obj_pose_params, cfg, opt_mode, exposure_params)
                pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)
                pos_grad_count = torch.zeros(scene.num_gaussians, device=device)
                print(f"  [densify] N={scene.num_gaussians}")

            if do_densify and cfg.reset_opacity_every > 0 and (i + 1) % cfg.reset_opacity_every == 0:
                with torch.no_grad():
                    # sigmoid(-4) ≈ 0.018 — small but nonzero so Gaussians aren't
                    # immediately pruned; they must re-earn opacity from the loss.
                    scene._opacities_raw.data.fill_(-4.0)
                optimizer = _setup_optimizer(scene, body_pose_params, obj_pose_params, cfg, opt_mode, exposure_params)
                print(f"  [opacity reset] iter {i+1}")

            if (i + 1) % 500 == 0 or i == n_iters - 1:
                _checkpoint(f"{log_label.lower().replace(' ', '_')}_iter{i+1:05d}", batch_frames[0])

    for cycle in range(cfg.n_cycles):
        c_label = f"C{cycle+1}of{cfg.n_cycles}"

        if cfg.alternating:
            n_canon = cfg.iterations_canonical_per_cycle
            print(f"\n--- cycle {cycle+1}/{cfg.n_cycles}: canonical ({n_canon} iters, densification ON) ---")
            _run_cycle_phase('canonical', n_canon, f"{c_label} canon", do_densify=True)

            n_pose = cfg.iterations_pose_per_cycle
            print(f"\n--- cycle {cycle+1}/{cfg.n_cycles}: pose ({n_pose} iters) ---")
            _run_cycle_phase('pose', n_pose, f"{c_label} pose", do_densify=False)
        else:
            n_joint = cfg.iterations_canonical_per_cycle + cfg.iterations_pose_per_cycle
            print(f"\n--- cycle {cycle+1}/{cfg.n_cycles}: joint ({n_joint} iters, densification ON) ---")
            _run_cycle_phase('joint', n_joint, f"{c_label} joint", do_densify=True)

    # ------------------------------------------------------------------ #
    # Final refinement phase (all params, 0.1× LR)
    # ------------------------------------------------------------------ #
    n_iters = cfg.iterations_refine
    optimizer = _setup_optimizer(scene, body_pose_params, obj_pose_params, cfg, 'refine', exposure_params)
    print(f"\n--- Refinement ({n_iters} iters, 0.1× LR) ---")

    for i in range(n_iters):
        batch_frames = _sample_batch()
        compute_entity_mask = (
            cfg.loss_weights.get('entity_mask', 1.0) > 0
            and (i % cfg.entity_mask_interval == 0)
        )
        optimizer.zero_grad()
        batch_loss, terms = _compute_batch_loss(
            scene, batch_frames, compute_entity_mask=compute_entity_mask,
            **_loss_kwargs
        )
        batch_loss.backward()
        optimizer.step()
        total_iter += 1
        _log("[refine]", i, n_iters, batch_loss.item(), terms, scene.num_gaussians)

        if (i + 1) % 500 == 0 or i == n_iters - 1:
            _checkpoint(f"refine_iter{i+1:05d}", batch_frames[0])

    # ------------------------------------------------------------------ #
    # Final pose sweep: canonical frozen, every frame updated once per pass.
    # Catches frames that were never sampled due to frame_step > 1, and gives
    # a final per-frame polish against the converged canonical Gaussians.
    # ------------------------------------------------------------------ #
    if cfg.n_pose_sweep_passes > 0:
        # Extend the data cache for any frames not in the training set.
        # The existing caches are dicts and _loss_kwargs holds references to
        # them, so additions here are visible to _compute_batch_loss.
        missing = [t for t in all_frame_indices if t not in _cache_rgb]
        if missing:
            print(f"  [pose sweep] Loading {len(missing)} additional frames...", flush=True)
            for t in missing:
                _cache_rgb[t] = _load_target_rgb(video_path, t, H_tr, W_tr, device)
                raw_d = _load_depth_tensor(depth_folder, t, device)
                _cache_depth[t] = _resize_to_tr(raw_d) if raw_d is not None else None
                for oid in all_oids:
                    raw_m = _load_mask_tensor(masks_dir, oid, t, device)
                    if raw_m is not None and oid in human_oids:
                        raw_m = _fill_body_mask(raw_m)
                    _cache_masks[(oid, t)] = _resize_to_tr(raw_m) if raw_m is not None else None

        optimizer_sweep = _setup_optimizer(
            scene, body_pose_params, obj_pose_params, cfg, 'pose', exposure_params
        )
        n_frames = len(all_frame_indices)
        print(f"\n--- Pose sweep: {cfg.n_pose_sweep_passes} pass(es) × {n_frames} frames ---")

        for pass_idx in range(cfg.n_pose_sweep_passes):
            for fi, frame_t in enumerate(all_frame_indices):
                optimizer_sweep.zero_grad()
                batch_loss, terms = _compute_batch_loss(
                    scene, [frame_t], compute_entity_mask=True, **_loss_kwargs
                )
                batch_loss.backward()
                optimizer_sweep.step()

                total_steps = cfg.n_pose_sweep_passes * n_frames
                step = pass_idx * n_frames + fi + 1
                log_every = max(1, n_frames // 10)
                if step % log_every == 0 or step == total_steps:
                    terms_str = ' '.join(f"{k}={v:.4f}" for k, v in terms.items())
                    print(f"  [sweep {pass_idx+1}/{cfg.n_pose_sweep_passes}]"
                          f"[{fi+1}/{n_frames}] loss={batch_loss.item():.4f}  {terms_str}")

    return scene


def _load_target_rgb(video_path: str, frame_idx: int, H: int, W: int, device: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return torch.zeros(H, W, 3, device=device)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Resize to match depth resolution if needed
    if rgb.shape[0] != H or rgb.shape[1] != W:
        rgb = cv2.resize(rgb, (W, H))
    return torch.tensor(rgb, dtype=torch.float32, device=device)


def _render_np(scene, world_pos, viewmat, K, H, W, sh_degree, device) -> np.ndarray:
    """Render one view; return (H, W, 3) uint8 numpy array."""
    result = render(scene, world_pos, viewmat, K, H, W, sh_degree=sh_degree)
    return (result.rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
