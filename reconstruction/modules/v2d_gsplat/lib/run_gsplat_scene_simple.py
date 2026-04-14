"""
Simplified joint 4D Gaussian Splatting optimization.

All parameters (Gaussian geometry, body pose, object SE(3)) are optimized
jointly in a single flat training loop with no alternating phases, no
hard-negative mining, and no confidence gating.  Intended for debugging and
rapid experimentation.

Inputs / outputs mirror run_video_to_gsplat.py; the simplification is purely
in the training loop.
"""

import math
import os
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.gsplat.lib.scene import GaussianScene, ENTITY_BODY, ENTITY_OBJECT_BASE
from v2d.gsplat.lib.deformation import (
    SmplDeformer, BodyPoseParams, ObjectPoseParams, rotation_6d_to_matrix,
)
from v2d.gsplat.lib.initialization import build_scene
from v2d.gsplat.lib.optimization import (
    compute_world_positions, _setup_optimizer, _load_frame_rgb,
    _load_depth_tensor, _load_mask_tensor,
)
from v2d.gsplat.lib.rasterizer import render, render_semantic, build_viewmat, build_K
from v2d.gsplat.lib.losses import loss_rgb, loss_ssim
from v2d.gsplat.lib.densification import densify_and_prune
from v2d.gsplat.lib.extraction import save_gaussians_ply, save_smpl_results, save_object_poses


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SceneSimpleConfig:
    n_batches: int = 10000
    batch_size: int = 4
    render_interval: int = 500   # save debug renders every N batches

    # Loss weights
    weight_rgb: float = 1.0
    weight_ssim: float = 0.2
    weight_depth: float = 0.1
    weight_semantic: float = 1.0   # multi-class focal loss over depth-composited class render
    semantic_focal_gamma: float = 2.0  # focal loss γ; 0 = plain weighted CE
    weight_obj_anchor: float = 0.0     # pull object canonical positions toward mesh-init reference

    # Pose smoothness (SE(3) / body joints)
    weight_obj_smooth: float = 0.0
    weight_body_smooth: float = 0.0

    # Gaussian learning rates
    lr_positions: float = 1.6e-4
    lr_rotations: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 1e-3
    lr_sh_rest: float = 1e-4
    lr_skinning: float = 1e-4

    # Body pose learning rates
    lr_body_pose: float = 1e-3      # global orient
    lr_body_joints: float = 0.0     # joint angles (0 = lock to NLF)
    lr_body_shape: float = 1e-4
    lr_body_transl: float = 1e-3

    # Object pose learning rates
    lr_obj_pose: float = 1e-3
    lr_obj_scale: float = 1e-4

    # Global LR scale — multiplies all learning rates; useful for quick experiments
    lr_scale: float = 1.0

    # LR decay
    lr_decay_schedule: str = 'cosine'   # 'cosine' | 'exponential' | 'none'
    lr_decay_final: float = 0.1

    # Densification
    densify_every: int = 500
    grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.005
    max_gaussians: int = 500_000
    max_scale_factor: float = 0.1
    reset_opacity_every: int = 10_000_000   # effectively disabled

    # Scene
    sh_degree: int = 1
    train_scale: float = 0.5
    initial_opacity_obj: float = 0.05
    body_subdivisions: int = 0

    device: str = 'cuda'


def load_config(config_path: Optional[str]) -> SceneSimpleConfig:
    cfg = SceneSimpleConfig()
    if not config_path or not os.path.exists(config_path):
        return cfg
    import yaml
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f"[scene_simple] WARNING: unknown config key '{k}' — ignored")
    return cfg


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _load_target_rgb(
    video_path: str, frame_idx: int, H: int, W: int, device: str,
) -> torch.Tensor:
    rgb = _load_frame_rgb(video_path, frame_idx).astype(np.float32) / 255.0
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    return torch.tensor(rgb, dtype=torch.float32, device=device)


def _preload_frame_data(
    video_path: str,
    depth_folder: str,
    masks_dir: str,
    frame_indices: List[int],
    all_oids: List[int],
    H: int, W: int,
    device: str,
) -> Tuple[Dict, Dict, Dict]:
    """
    Pre-load all frames into GPU memory.
    Returns:
      rgb_cache:   {frame_idx: (H, W, 3) tensor}
      depth_cache: {frame_idx: (H, W) tensor}  — None entries skipped
      mask_cache:  {(oid, frame_idx): (H, W) tensor}  — None entries skipped
    """
    print(f"[scene_simple] Pre-loading {len(frame_indices)} frames…")
    rgb_cache: Dict[int, torch.Tensor] = {}
    depth_cache: Dict[int, torch.Tensor] = {}
    mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    for t in frame_indices:
        rgb_cache[t] = _load_target_rgb(video_path, t, H, W, device)
        d = _load_depth_tensor(depth_folder, t, device)
        if d is not None:
            if d.shape != (H, W):
                d_np = d.cpu().numpy()
                d_np = cv2.resize(d_np, (W, H), interpolation=cv2.INTER_NEAREST)
                d = torch.tensor(d_np, dtype=torch.float32, device=device)
            depth_cache[t] = d
        for oid in all_oids:
            m = _load_mask_tensor(masks_dir, oid, t, device)
            if m is not None:
                if m.shape != (H, W):
                    m_np = m.cpu().numpy()
                    m_np = cv2.resize(m_np, (W, H), interpolation=cv2.INTER_NEAREST)
                    m = torch.tensor(m_np, dtype=torch.float32, device=device)
                mask_cache[(oid, t)] = m

    print(f"[scene_simple]   RGB: {len(rgb_cache)}  depth: {len(depth_cache)}  masks: {len(mask_cache)}")
    return rgb_cache, depth_cache, mask_cache


# --------------------------------------------------------------------------- #
# LR decay
# --------------------------------------------------------------------------- #

def _apply_lr_decay(optimizer: torch.optim.Adam, step: int, total: int, cfg: SceneSimpleConfig):
    if cfg.lr_decay_schedule == 'none':
        return
    if cfg.lr_decay_schedule == 'cosine':
        factor = cfg.lr_decay_final + 0.5 * (1.0 - cfg.lr_decay_final) * (1 + math.cos(math.pi * step / total))
    else:  # exponential
        factor = cfg.lr_decay_final ** (step / total)
    for pg in optimizer.param_groups:
        pg['lr'] = pg.get('_base_lr', pg['lr']) * factor


# --------------------------------------------------------------------------- #
# Semantic class rendering helpers
# --------------------------------------------------------------------------- #

def _build_entity_class_map(
    scene: GaussianScene,
    entity_role_map: Dict[int, str],
) -> Tuple[torch.Tensor, Dict[int, int], int]:
    """
    Build a compact class index for each Gaussian and a mapping
    {scene_entity_id → class_index}.

    Class layout:
      0           = background
      1           = human body  (ENTITY_BODY)
      2, 3, ...   = objects     (ENTITY_OBJECT_BASE + rid), in rid order
    """
    object_oids = sorted(oid for oid, r in entity_role_map.items() if r == 'object')
    entity_to_class: Dict[int, int] = {0: 0}                          # background
    if any(r == 'human' for r in entity_role_map.values()):
        entity_to_class[ENTITY_BODY] = 1
    for rid, _oid in enumerate(object_oids):
        entity_to_class[ENTITY_OBJECT_BASE + rid] = 2 + rid

    n_classes = max(entity_to_class.values()) + 1

    class_map = torch.zeros(scene.num_gaussians, dtype=torch.long,
                            device=scene.entity_ids.device)
    for eid, cidx in entity_to_class.items():
        class_map[scene.entity_ids == eid] = cidx

    return class_map, entity_to_class, n_classes


def _build_gt_label_map(
    masks: Dict[int, torch.Tensor],   # {oid: (H, W) float {0,1}}
    entity_role_map: Dict[int, str],
    entity_to_class: Dict[int, int],  # scene_entity_id → class_index
    H: int, W: int,
    device: str,
) -> torch.Tensor:                    # (H, W) long, class indices
    """
    Merge per-entity SAM2 masks into a single GT label map.

    Objects are painted first; the human mask is applied last so it takes
    priority at any overlap (the person is the occluder).
    """
    object_oids = sorted(oid for oid, r in entity_role_map.items() if r == 'object')
    human_oids  = [oid for oid, r in entity_role_map.items() if r == 'human']

    label_map = torch.zeros(H, W, dtype=torch.long, device=device)

    for rid, oid in enumerate(object_oids):
        m = masks.get(oid)
        if m is not None:
            cidx = entity_to_class.get(ENTITY_OBJECT_BASE + rid, 0)
            label_map[m > 0.5] = cidx

    for oid in human_oids:
        m = masks.get(oid)
        if m is not None:
            cidx = entity_to_class.get(ENTITY_BODY, 0)
            label_map[m > 0.5] = cidx

    return label_map


# --------------------------------------------------------------------------- #
# Per-frame loss
# --------------------------------------------------------------------------- #

def _compute_frame_loss(
    scene: GaussianScene,
    world_pos: torch.Tensor,
    rgb: torch.Tensor,
    depth: Optional[torch.Tensor],
    masks: Dict[int, torch.Tensor],   # {oid: (H, W)}
    entity_role_map: Dict[int, str],
    entity_class_map: torch.Tensor,   # (N,) long — pre-built, reused each step
    entity_to_class: Dict[int, int],
    n_classes: int,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int, W: int,
    cfg: SceneSimpleConfig,
) -> Tuple[torch.Tensor, dict]:
    result = render(scene, world_pos, viewmat, K, H, W, sh_degree=cfg.sh_degree)

    rgb_loss  = loss_rgb(result.rgb, rgb)
    ssim_loss = loss_ssim(result.rgb, rgb)
    total = cfg.weight_rgb * rgb_loss + cfg.weight_ssim * ssim_loss
    terms = {'rgb': float(rgb_loss), 'ssim': float(ssim_loss)}

    # Depth
    if cfg.weight_depth > 0 and depth is not None:
        valid = depth > 0
        if valid.any():
            depth_loss = F.l1_loss(result.depth[valid], depth[valid])
            total = total + cfg.weight_depth * depth_loss
            terms['depth'] = float(depth_loss)

    # Semantic segmentation: render class labels through full depth-ordered scene,
    # compare to merged GT label map via cross-entropy.
    # Occlusion is handled naturally: occluded Gaussians contribute less to the
    # composited class output because closer Gaussians accumulate alpha first.
    if cfg.weight_semantic > 0 and masks:
        gt_labels = _build_gt_label_map(
            masks, entity_role_map, entity_to_class, H, W, str(rgb.device)
        )
        class_render = render_semantic(
            scene, world_pos, entity_class_map, n_classes, viewmat, K, H, W
        )  # (H, W, n_classes) — soft class probabilities via alpha compositing

        # Focal loss with inverse-frequency class weighting.
        # Class weights fix static imbalance (bg >> object pixels).
        # Focal modulation (1-p)^γ additionally down-weights easy well-classified
        # pixels so hard thin-structure regions dominate the gradient.
        logits = class_render.reshape(-1, n_classes)
        labels = gt_labels.reshape(-1)
        freq = torch.bincount(labels, minlength=n_classes).float().clamp(min=1)
        class_weights = (labels.numel() / (n_classes * freq)).clamp(max=10.0)
        # Per-pixel CE (unreduced) with class weights
        ce = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
        if cfg.semantic_focal_gamma > 0:
            # p_t: model probability assigned to the correct class
            with torch.no_grad():
                p_t = F.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t) ** cfg.semantic_focal_gamma
            semantic_loss = (focal_weight * ce).mean()
        else:
            semantic_loss = ce.mean()
        total = total + cfg.weight_semantic * semantic_loss
        terms['semantic'] = float(semantic_loss)

    return total, terms


# --------------------------------------------------------------------------- #
# Pose smoothness
# --------------------------------------------------------------------------- #

def _obj_smoothness_loss(
    obj_pose_params: ObjectPoseParams,
    frame_indices: List[int],
) -> torch.Tensor:
    sorted_f = sorted(frame_indices)
    if len(sorted_f) < 2:
        return torch.tensor(0.0, device=obj_pose_params.translations.device)
    t_curr = torch.stack([obj_pose_params.translations[t, 0] for t in sorted_f[:-1]])
    t_next = torch.stack([obj_pose_params.translations[t, 0] for t in sorted_f[1:]])
    r_curr = rotation_6d_to_matrix(
        torch.stack([obj_pose_params.rotations_6d[t, 0] for t in sorted_f[:-1]])
    )
    r_next = rotation_6d_to_matrix(
        torch.stack([obj_pose_params.rotations_6d[t, 0] for t in sorted_f[1:]])
    )
    return F.mse_loss(t_curr, t_next) + F.mse_loss(r_curr, r_next)


def _body_smoothness_loss(
    body_pose_params: BodyPoseParams,
    frame_indices: List[int],
) -> torch.Tensor:
    sorted_f = sorted(frame_indices)
    if len(sorted_f) < 2:
        return torch.tensor(0.0, device=body_pose_params.transl.device)
    t_curr = torch.stack([body_pose_params.transl[t] for t in sorted_f[:-1]])
    t_next = torch.stack([body_pose_params.transl[t] for t in sorted_f[1:]])
    return F.mse_loss(t_curr, t_next)


# --------------------------------------------------------------------------- #
# Debug renders
# --------------------------------------------------------------------------- #

# Fixed BGR palette: background=dark gray, human=blue, objects=red/green/yellow/...
_CLASS_COLORS_BGR = [
    (40,  40,  40),   # 0 background
    (200, 80,  20),   # 1 human  (orange-blue in BGR)
    (30,  30,  220),  # 2 object 0 (red)
    (30,  180, 30),   # 3 object 1 (green)
    (0,   200, 200),  # 4 object 2 (yellow)
    (200, 30,  200),  # 5 object 3 (magenta)
]


def _label_map_to_bgr(label_map: np.ndarray) -> np.ndarray:
    """Convert (H, W) int class map to (H, W, 3) BGR image."""
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cidx, color in enumerate(_CLASS_COLORS_BGR):
        out[label_map == cidx] = color
    return out


def _save_debug_renders(
    scene, body_pose_params, obj_pose_params, smpl_deformer,
    rgb_cache, mask_cache, all_oids, frame_indices,
    entity_class_map, entity_to_class, entity_role_map, n_classes,
    viewmat, K, H, W,
    batch_idx: int, output_dir: str, cfg: SceneSimpleConfig,
    n_frames: int = 4,
) -> None:
    debug_dir = os.path.join(output_dir, 'debug', f'batch_{batch_idx:06d}')
    os.makedirs(debug_dir, exist_ok=True)
    sample = frame_indices[::max(1, len(frame_indices) // n_frames)][:n_frames]
    with torch.no_grad():
        for t in sample:
            rgb = rgb_cache[t]
            world_pos = compute_world_positions(
                scene, body_pose_params, obj_pose_params, smpl_deformer, t
            )
            result = render(scene, world_pos, viewmat, K, H, W, sh_degree=cfg.sh_degree)
            rendered_np = (result.rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            original_np = (rgb.cpu().numpy() * 255).astype(np.uint8)

            # Rendered semantic: argmax over class dimension
            class_render = render_semantic(
                scene, world_pos, entity_class_map, n_classes, viewmat, K, H, W
            )
            pred_labels = class_render.argmax(dim=-1).cpu().numpy()  # (H, W)
            pred_sem_bgr = _label_map_to_bgr(pred_labels)

            # GT semantic from masks
            masks_t = {oid: mask_cache[(oid, t)] for oid in all_oids if (oid, t) in mask_cache}
            gt_label_map = _build_gt_label_map(
                masks_t, entity_role_map, entity_to_class, H, W, str(rgb.device)
            )
            gt_sem_bgr = _label_map_to_bgr(gt_label_map.cpu().numpy())

            combined = np.concatenate([
                cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR),
                gt_sem_bgr,
                pred_sem_bgr,
            ], axis=1)
            cv2.putText(combined, f'batch {batch_idx}  frame {t}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, f"{t:06d}.png"), combined)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def gsplat_scene_simple(
    video_path: str,
    depth_folder: str,
    intrinsics_path: str,
    masks_dir: str,
    prompts_path: str,
    output_dir: str,
    weights_dir: str,
    config_path: Optional[str] = None,
    smpl_path: Optional[str] = None,
    object_meshes_dir: Optional[str] = None,
    num_frames: Optional[int] = None,
    frame_step: int = 1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    device = cfg.device

    # ------------------------------------------------------------------ #
    # Parse inputs
    # ------------------------------------------------------------------ #
    intrinsics = CameraIntrinsics.load(intrinsics_path)
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    entity_role_map: Dict[int, str] = {}
    for p in prompts_data.get('prompts', []):
        oid = int(p['object_id'])
        role = p.get('role', 'object').lower()
        entity_role_map[oid] = role if role in ('human', 'object') else 'object'

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if num_frames is not None:
        total_frames = min(num_frames, total_frames)
    frame_indices = list(range(0, total_frames, max(1, frame_step)))

    object_oids = sorted(oid for oid, r in entity_role_map.items() if r == 'object')
    human_oids  = sorted(oid for oid, r in entity_role_map.items() if r == 'human')
    all_oids    = sorted(entity_role_map.keys())
    has_human   = bool(human_oids)

    print(f"[scene_simple] {len(frame_indices)} frames  entities={entity_role_map}  device={device}")

    # Scan object assets
    object_mesh_paths: Dict[int, str] = {}
    object_transform_paths: Dict[int, str] = {}
    object_fp_poses_dirs: Dict[int, str] = {}
    if object_meshes_dir and os.path.isdir(object_meshes_dir):
        for oid in object_oids:
            for attr, fname in [('mesh', f'object_{oid}.obj'),
                                 ('xform', f'object_{oid}_transform.json'),
                                 ('poses', f'object_{oid}_fp_poses')]:
                cand = os.path.join(object_meshes_dir, fname)
                if os.path.exists(cand):
                    {'mesh': object_mesh_paths, 'xform': object_transform_paths,
                     'poses': object_fp_poses_dirs}[attr][oid] = cand

    # ------------------------------------------------------------------ #
    # SMPL + body pose
    # ------------------------------------------------------------------ #
    smpl_deformer = None
    body_pose_params = None
    smpl_model_type = 'smpl'
    smpl_gender = 'neutral'

    if has_human and os.path.isdir(weights_dir):
        if smpl_path and os.path.exists(smpl_path):
            import h5py
            if h5py.is_hdf5(smpl_path):
                with h5py.File(smpl_path, 'r') as f:
                    _dec = lambda v: v.decode() if isinstance(v, bytes) else str(v)
                    smpl_model_type = _dec(f['model_type'][()])
                    smpl_gender     = _dec(f['gender'][()])
            else:
                meta = np.load(smpl_path, allow_pickle=True)
                smpl_model_type = str(meta.get('model_type', 'smpl'))
                smpl_gender     = str(meta.get('gender', 'neutral'))
        try:
            smpl_deformer = SmplDeformer(
                weights_dir, gender=smpl_gender, model_type=smpl_model_type, device=device,
            )
            n_body_joints = smpl_deformer.body_model.NUM_BODY_JOINTS
            body_pose_params = BodyPoseParams(total_frames, n_body_joints, device=device)
            if smpl_path and os.path.exists(smpl_path):
                body_pose_params.load_from_npz(smpl_path)
            body_pose_params = body_pose_params.to(device)
            print(f"[scene_simple] SMPL loaded ({smpl_model_type}, {smpl_gender})")
        except Exception as e:
            print(f"[scene_simple] WARNING: could not load SMPL: {e}")
            smpl_deformer = body_pose_params = None

    # ------------------------------------------------------------------ #
    # Object pose params
    # ------------------------------------------------------------------ #
    obj_pose_params = None
    if object_oids:
        obj_pose_params = ObjectPoseParams(total_frames, len(object_oids), device=device)
        for rid, oid in enumerate(object_oids):
            poses_dir = object_fp_poses_dirs.get(oid)
            xform_path = object_transform_paths.get(oid)
            if poses_dir and xform_path:
                obj_pose_params.load_from_fp_poses_dir(poses_dir, rid, xform_path)

    # ------------------------------------------------------------------ #
    # Scene initialisation
    # ------------------------------------------------------------------ #
    smpl_betas = body_pose_params.betas.detach() if body_pose_params is not None else None
    scene = build_scene(
        video_path=video_path,
        depth_folder=depth_folder,
        intrinsics=intrinsics,
        masks_dir=masks_dir,
        entity_role_map=entity_role_map,
        smpl_deformer=smpl_deformer,
        smpl_betas=smpl_betas,
        object_mesh_paths=object_mesh_paths,
        object_transform_paths=object_transform_paths,
        initial_opacity_obj=cfg.initial_opacity_obj,
        body_subdivisions=cfg.body_subdivisions,
        device=device,
    ).to(device)

    # ------------------------------------------------------------------ #
    # Frame data
    # ------------------------------------------------------------------ #
    s = cfg.train_scale
    intrinsics_tr = CameraIntrinsics(
        fx=intrinsics.fx * s, fy=intrinsics.fy * s,
        cx=intrinsics.cx * s, cy=intrinsics.cy * s,
        width=int(intrinsics.width * s), height=int(intrinsics.height * s),
    )
    H, W = intrinsics_tr.height, intrinsics_tr.width
    viewmat = build_viewmat(device)
    K = build_K(intrinsics_tr, device)

    rgb_cache, depth_cache, mask_cache = _preload_frame_data(
        video_path, depth_folder, masks_dir, frame_indices, all_oids, H, W, device
    )

    # ------------------------------------------------------------------ #
    # Optimizer (always joint — no alternating phases)
    # ------------------------------------------------------------------ #
    from v2d.gsplat.lib.optimization import OptimConfig
    s = cfg.lr_scale
    optim_cfg = OptimConfig(
        lr_positions=cfg.lr_positions * s, lr_rotations=cfg.lr_rotations * s,
        lr_scales=cfg.lr_scales * s, lr_opacities=cfg.lr_opacities * s,
        lr_sh_dc=cfg.lr_sh_dc * s, lr_sh_rest=cfg.lr_sh_rest * s,
        lr_skinning=cfg.lr_skinning * s, lr_body_pose=cfg.lr_body_pose * s,
        lr_body_joints=cfg.lr_body_joints * s, lr_body_shape=cfg.lr_body_shape * s,
        lr_body_transl=cfg.lr_body_transl * s, lr_obj_pose=cfg.lr_obj_pose * s,
        lr_obj_scale=cfg.lr_obj_scale * s, device=device,
    )
    optimizer = _setup_optimizer(scene, body_pose_params, obj_pose_params, optim_cfg, mode='joint')
    pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)

    entity_class_map, entity_to_class, n_classes = _build_entity_class_map(scene, entity_role_map)

    # Anchor object Gaussians to mesh-init canonical positions to resist pose-noise drift
    obj_anchor_positions = None
    if cfg.weight_obj_anchor > 0 and object_oids:
        obj_mask = scene.entity_ids >= ENTITY_OBJECT_BASE
        obj_anchor_positions = scene.positions[obj_mask].detach().clone()

    print(f"[scene_simple] Training for {cfg.n_batches} batches…\n")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    for i in range(cfg.n_batches):
        batch = random.sample(frame_indices, min(cfg.batch_size, len(frame_indices)))

        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        batch_terms: dict = {}

        for t in batch:
            rgb   = rgb_cache[t]
            depth = depth_cache.get(t)
            masks = {oid: mask_cache[(oid, t)] for oid in all_oids if (oid, t) in mask_cache}

            world_pos = compute_world_positions(
                scene, body_pose_params, obj_pose_params, smpl_deformer, t
            )
            loss, terms = _compute_frame_loss(
                scene, world_pos, rgb, depth, masks, entity_role_map,
                entity_class_map, entity_to_class, n_classes,
                viewmat, K, H, W, cfg,
            )
            batch_loss = batch_loss + loss / len(batch)
            for k, v in terms.items():
                batch_terms[k] = batch_terms.get(k, 0.0) + v / len(batch)

        # Pose smoothness over all frames
        if cfg.weight_obj_smooth > 0 and obj_pose_params is not None:
            smooth = _obj_smoothness_loss(obj_pose_params, frame_indices)
            batch_loss = batch_loss + cfg.weight_obj_smooth * smooth
            batch_terms['obj_smooth'] = float(smooth)

        if cfg.weight_body_smooth > 0 and body_pose_params is not None:
            smooth = _body_smoothness_loss(body_pose_params, frame_indices)
            batch_loss = batch_loss + cfg.weight_body_smooth * smooth
            batch_terms['body_smooth'] = float(smooth)

        if cfg.weight_obj_anchor > 0 and obj_anchor_positions is not None:
            obj_mask = scene.entity_ids >= ENTITY_OBJECT_BASE
            obj_pos = scene.positions[obj_mask]
            if obj_pos.shape[0] == obj_anchor_positions.shape[0]:
                anchor_loss = F.mse_loss(obj_pos, obj_anchor_positions)
                batch_loss = batch_loss + cfg.weight_obj_anchor * anchor_loss
                batch_terms['obj_anchor'] = float(anchor_loss)

        batch_loss.backward()

        if scene._positions.grad is not None:
            pos_grad_accum += scene._positions.grad.norm(dim=-1).detach()

        optimizer.step()
        _apply_lr_decay(optimizer, i, cfg.n_batches, cfg)

        # Opacity reset
        if cfg.reset_opacity_every > 0 and (i + 1) % cfg.reset_opacity_every == 0:
            with torch.no_grad():
                scene._opacities_raw.data.fill_(-3.0)

        # Densification
        if (i + 1) % cfg.densify_every == 0 and i < cfg.n_batches - 1:
            extent = float(scene.positions.norm(dim=-1).max())
            scene = densify_and_prune(
                scene, pos_grad_accum,
                grad_threshold=cfg.grad_threshold,
                prune_opacity_threshold=cfg.prune_opacity_threshold,
                max_scene_extent=extent,
                max_gaussians=cfg.max_gaussians,
                max_scale_factor=cfg.max_scale_factor,
            )
            pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)
            optimizer = _setup_optimizer(
                scene, body_pose_params, obj_pose_params, optim_cfg, mode='joint'
            )
            entity_class_map, entity_to_class, n_classes = _build_entity_class_map(scene, entity_role_map)
            if cfg.weight_obj_anchor > 0 and object_oids:
                obj_mask = scene.entity_ids >= ENTITY_OBJECT_BASE
                obj_anchor_positions = scene.positions[obj_mask].detach().clone()

        if (i + 1) % cfg.render_interval == 0 or i == 0:
            terms_str = '  '.join(f"{k}={v:.4f}" for k, v in batch_terms.items())
            print(f"  batch {i+1:>5}/{cfg.n_batches}  loss={batch_loss.item():.4f}  "
                  f"{terms_str}  N={scene.num_gaussians}")
            _save_debug_renders(
                scene, body_pose_params, obj_pose_params, smpl_deformer,
                rgb_cache, mask_cache, all_oids, frame_indices,
                entity_class_map, entity_to_class, entity_role_map, n_classes,
                viewmat, K, H, W,
                i + 1, output_dir, cfg,
            )

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    print("\n[scene_simple] Saving outputs…")
    save_gaussians_ply(scene, os.path.join(output_dir, 'gaussians.ply'))
    if body_pose_params is not None:
        save_smpl_results(body_pose_params, os.path.join(output_dir, 'poses'),
                          model_type=smpl_model_type, gender=smpl_gender)
    if obj_pose_params is not None:
        save_object_poses(obj_pose_params, os.path.join(output_dir, 'poses'))
    print(f"[scene_simple] Done.  Outputs at: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplified joint 4D Gaussian Splatting')
    parser.add_argument('--video_path',        required=True)
    parser.add_argument('--depth_folder',      required=True)
    parser.add_argument('--intrinsics_path',   required=True)
    parser.add_argument('--masks_dir',         required=True)
    parser.add_argument('--prompts_path',      required=True)
    parser.add_argument('--output_dir',        required=True)
    parser.add_argument('--weights_dir',       required=True)
    parser.add_argument('--config_path',       default=None)
    parser.add_argument('--smpl_path',         default=None)
    parser.add_argument('--object_meshes_dir', default=None)
    parser.add_argument('--num_frames',        type=int, default=None)
    parser.add_argument('--frame_step',        type=int, default=1)
    args = parser.parse_args()

    gsplat_scene_simple(
        video_path=args.video_path,
        depth_folder=args.depth_folder,
        intrinsics_path=args.intrinsics_path,
        masks_dir=args.masks_dir,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        config_path=args.config_path,
        smpl_path=args.smpl_path,
        object_meshes_dir=args.object_meshes_dir,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
    )
