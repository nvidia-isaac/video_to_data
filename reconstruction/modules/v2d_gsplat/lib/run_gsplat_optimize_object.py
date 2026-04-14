"""
Standalone rigid-object Gaussian optimization.

Jointly optimizes canonical Gaussian shape and per-frame SE(3) poses for a
single rigid object.  Loss is masked RGB + SSIM computed only within the object
silhouette mask.  No body/SMPL, no background, no alternating phases.

Inputs:
  images_dir      - folder of RGB images ({frame:06d}.png)
  masks_dir       - folder of object masks ({frame:06d}.png)
  intrinsics_path - CameraIntrinsics JSON
  output_dir      - root output directory

Optional inputs:
  config_path  - YAML overriding any ObjectOptimConfig field
  mesh_path    - .obj for Gaussian init (falls back to depth unproject)
  depth_dir    - depth folder used only for mesh alignment at init
  poses_dir    - per-frame initial SE(3) JSONs ({frame:06d}.json, Transform3d format)
                 Frame-0 JSON defines canonical; others are made relative to it.

Outputs written to output_dir/:
  gaussians.ply                    - optimised canonical Gaussians
  poses/{frame:06d}.json           - refined per-frame SE(3) (Transform3d format)
  debug/batch_{N:06d}/{t:06d}.png  - render-vs-original snapshots at render_interval
"""

import math
import os
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.gsplat.lib.scene import GaussianScene, FeatureGaussians, ENTITY_OBJECT_BASE
from v2d.gsplat.lib.deformation import ObjectPoseParams, rotation_6d_to_matrix
from v2d.gsplat.lib.initialization import init_object_from_mesh, _init_object_from_depth
from v2d.gsplat.lib.rasterizer import render, build_viewmat, build_K
from v2d.gsplat.lib.losses import loss_ssim, loss_anchor, loss_mask_asymmetric
from v2d.gsplat.lib.densification import densify_and_prune
from v2d.gsplat.lib.extraction import save_gaussians_ply


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class ObjectOptimConfig:
    n_batches: int = 3000
    batch_size: int = 4
    render_interval: int = 200   # save debug renders every N batches

    # Loss weights (masked to object silhouette)
    weight_rgb: float = 1.0
    weight_ssim: float = 0.2

    # Gaussian learning rates
    lr_positions: float = 1.6e-4
    lr_rotations: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 1e-3
    lr_sh_rest: float = 1e-4

    # Mask (silhouette): asymmetric focal BCE between rendered alpha and object mask
    weight_mask: float = 0.0
    mask_outside_weight: float = 0.1      # penalty outside object mask (free background)
    mask_occluder_weight: float = 0.0     # penalty outside object mask but inside person mask
                                          # (object may be there but hidden — default no penalty)
    mask_focal_gamma: float = 2.0         # focal loss γ; 0 = plain weighted BCE

    # Anchor: pull canonical positions back toward mesh-init reference
    weight_anchor: float = 0.0

    # Pose smoothness: penalize SE(3) change between consecutive frames
    weight_smoothness: float = 0.0

    # Centroid: pull projected Gaussian centroid toward mask centroid (2D translation anchor)
    weight_centroid: float = 0.0

    # Normal: cosine similarity between normals derived from rendered depth and input depth
    weight_normal: float = 0.0

    # Pose learning rate (SE(3) per frame)
    lr_pose: float = 1e-3

    # Global LR scale — multiplies all learning rates; useful for quick experiments
    lr_scale: float = 1.0

    # Progressive mode: process frames in order, alternating canonical/pose per frame
    progressive: bool = False
    # Plateau detection — each phase runs until loss stops improving (or max steps hit)
    progressive_min_steps: int = 10          # minimum steps before plateau check
    progressive_max_steps: int = 200         # hard cap per phase per inner iter
    progressive_patience: int = 20           # steps to look back when checking improvement
    progressive_plateau_threshold: float = 1e-4  # min relative improvement to continue
    progressive_ema_alpha: float = 0.3       # EMA smoothing for canonical loss (noisier)

    # LR decay over all n_batches
    lr_decay_schedule: str = 'cosine'   # 'cosine' | 'exponential' | 'none'
    lr_decay_final: float = 0.1         # final LR as fraction of initial

    # Densification
    densify_every: int = 100
    grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.005
    max_gaussians: int = 50_000
    max_scale_factor: float = 0.1
    reset_opacity_every: int = 500

    # Scene
    sh_degree: int = 3
    train_scale: float = 1.0   # resize images for faster training (e.g. 0.5 = half res)
    n_gaussians: int = 5_000   # target count when initializing from mesh
    initial_opacity: float = 0.1

    # Feature Gaussians (separate from RGB Gaussians; unconstrained opacities/scales)
    # Set feature_encoder='' to disable entirely.
    feature_encoder: str = 'dinov2_vits14'
    feature_proj_dim: int = 64          # projected feature dim; 0 = raw encoder dim
    feature_n_gaussians: int = 500      # fewer than RGB Gaussians
    weight_feature: float = 0.1
    lr_feat_positions: float = 1.6e-4
    lr_feat_rotations: float = 1e-3
    lr_feat_scales: float = 5e-3
    lr_feat_opacities: float = 5e-2
    lr_feat_features: float = 1e-3

    device: str = 'cuda'


def load_config(config_path: Optional[str]) -> ObjectOptimConfig:
    cfg = ObjectOptimConfig()
    if not config_path or not os.path.exists(config_path):
        return cfg
    import yaml
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f"[object_optim] WARNING: unknown config key '{k}' — ignored")
    return cfg


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _find_frame_indices(images_dir: str, frame_step: int) -> List[int]:
    indices = []
    for fname in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        try:
            indices.append(int(stem))
        except ValueError:
            continue
    return indices[::max(1, frame_step)]


def _load_rgb(path: str, scale: float, device: str) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return torch.tensor(img, dtype=torch.float32, device=device)


def _load_mask(path: str, scale: float, device: str) -> Optional[torch.Tensor]:
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if scale != 1.0:
        h, w = mask.shape
        mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    return torch.tensor((mask > 0.5).astype(np.float32), dtype=torch.float32, device=device)


def load_frame_data(
    images_dir: str,
    masks_dir: str,
    frame_indices: List[int],
    train_scale: float,
    device: str,
    person_masks_dir: Optional[str] = None,
) -> Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """Pre-load all frames. Returns {frame_idx: (rgb, obj_mask, person_mask)}."""
    data = {}
    for t in frame_indices:
        rgb_path    = os.path.join(images_dir, f"{t:06d}.png")
        mask_path   = os.path.join(masks_dir,  f"{t:06d}.png")
        if not os.path.exists(rgb_path):
            continue
        person_mask = None
        if person_masks_dir:
            person_mask = _load_mask(os.path.join(person_masks_dir, f"{t:06d}.png"),
                                     train_scale, device)
        data[t] = (
            _load_rgb(rgb_path, train_scale, device),
            _load_mask(mask_path, train_scale, device),
            person_mask,
        )
    return data


def _load_depth_frame(path: str, scale: float, device: str) -> Optional[torch.Tensor]:
    """Load uint16 inverse-depth PNG → metric depth float tensor (H, W)."""
    if not os.path.exists(path):
        return None
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # uint16
    if scale != 1.0:
        h, w = raw.shape
        raw = cv2.resize(raw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    # Decode: pixel = 65535 / (depth_m + 1)  →  depth_m = 65535 / pixel - 1
    valid = raw > 0
    depth = np.zeros_like(raw)
    depth[valid] = 65535.0 / raw[valid] - 1.0
    return torch.tensor(depth, dtype=torch.float32, device=device)


def _depth_to_normals(depth: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute surface normals from a metric depth map via finite differences.

    depth: (H, W) metric depth (0 = invalid)
    k:     (3, 3) camera intrinsics
    Returns (H, W, 3) unit normals in camera space; zero vector where invalid.
    """
    H, W = depth.shape
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]

    ys = torch.arange(H, device=depth.device, dtype=torch.float32)
    xs = torch.arange(W, device=depth.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    Z = depth
    X = (grid_x - cx) * Z / fx.clamp(min=1e-6)
    Y = (grid_y - cy) * Z / fy.clamp(min=1e-6)
    pts = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)

    # Central finite differences; edges stay zero
    dpdx = torch.zeros_like(pts)
    dpdy = torch.zeros_like(pts)
    dpdx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dpdy[1:-1, :] = pts[2:, :] - pts[:-2, :]

    normals = torch.linalg.cross(dpdx, dpdy)  # (H, W, 3)

    # Flip normals pointing away from camera (+Z in OpenCV convention)
    normals = torch.where(normals[..., 2:3] > 0, -normals, normals)

    norms = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    valid = (depth > 0).unsqueeze(-1)
    return torch.where(valid, normals / norms, torch.zeros_like(normals))


def load_depth_data(
    depth_dir: str,
    frame_indices: List[int],
    train_scale: float,
    device: str,
) -> Dict[int, torch.Tensor]:
    """Pre-load all depth frames. Returns {frame_idx: depth (H, W)}."""
    data = {}
    for t in frame_indices:
        d = _load_depth_frame(os.path.join(depth_dir, f"{t:06d}.png"), train_scale, device)
        if d is not None:
            data[t] = d
    return data


# --------------------------------------------------------------------------- #
# Pose initialization
# --------------------------------------------------------------------------- #

def _load_absolute_transform(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Transform3d JSON → (R: 3×3, t: 3) float64."""
    from scipy.spatial.transform import Rotation as SciRot
    from v2d.common.datatypes import Transform3d
    t3d = Transform3d.load(json_path)
    w, x, y, z = t3d.rotation
    R = SciRot.from_quat([x, y, z, w]).as_matrix()
    return R, np.array(t3d.translation, dtype=np.float64)


def init_poses(
    obj_pose_params: ObjectPoseParams,
    poses_dir: Optional[str],
    frame_indices: List[int],
) -> None:
    """Initialize pose params from per-frame Transform3d JSONs, relative to frame 0."""
    if not poses_dir or not os.path.isdir(poses_dir):
        return
    ref_path = os.path.join(poses_dir, f"{frame_indices[0]:06d}.json")
    if os.path.exists(ref_path):
        R0, t0 = _load_absolute_transform(ref_path)
    else:
        print(f"[object_optim] No frame-0 pose at {ref_path} — using identity as reference")
        R0, t0 = np.eye(3), np.zeros(3)

    loaded = 0
    for t in frame_indices:
        path = os.path.join(poses_dir, f"{t:06d}.json")
        if not os.path.exists(path):
            continue
        Rt, tt = _load_absolute_transform(path)
        R_rel = Rt @ R0.T
        t_rel = tt - R_rel @ t0
        r6d = np.concatenate([R_rel[:, 0], R_rel[:, 1]]).astype(np.float32)
        dev = obj_pose_params.rotations_6d.device
        with torch.no_grad():
            obj_pose_params.rotations_6d[t, 0] = torch.from_numpy(r6d).to(dev)
            obj_pose_params.translations[t, 0]  = torch.from_numpy(t_rel.astype(np.float32)).to(dev)
        loaded += 1
    print(f"[object_optim] Loaded {loaded} initial poses from {poses_dir}")

# --------------------------------------------------------------------------- #
# Scene initialization
# --------------------------------------------------------------------------- #

def init_scene(
    mesh_path: Optional[str],
    images_dir: str,
    depth_dir: Optional[str],
    masks_dir: str,
    intrinsics: CameraIntrinsics,
    frame_indices: List[int],
    poses_dir: Optional[str],
    cfg: ObjectOptimConfig,
) -> GaussianScene:
    device = cfg.device
    frame0 = frame_indices[0]
    depth_path0 = os.path.join(depth_dir, f"{frame0:06d}.png") if depth_dir else None
    mask_path0  = os.path.join(masks_dir, f"{frame0:06d}.png")

    mask0: Optional[np.ndarray] = None
    if os.path.exists(mask_path0):
        from PIL import Image
        m = np.array(Image.open(mask_path0))
        mask0 = (m[..., 0] if m.ndim == 3 else m) > 127

    transform_path0: Optional[str] = None
    if poses_dir and os.path.isdir(poses_dir):
        cand = os.path.join(poses_dir, f"{frame0:06d}.json")
        if os.path.exists(cand):
            transform_path0 = cand

    if mesh_path and os.path.exists(mesh_path):
        positions, colors = init_object_from_mesh(
            mesh_path=mesh_path,
            depth_path=depth_path0 or '',
            intrinsics=intrinsics,
            object_mask=mask0,
            transform_path=transform_path0,
            n_gaussians=cfg.n_gaussians,
            device=device,
        )
    elif depth_path0 and os.path.exists(depth_path0):
        print("[object_optim] No mesh provided — initializing from depth map")
        img0_path = os.path.join(images_dir, f"{frame0:06d}.png")
        frame0_rgb = (
            cv2.cvtColor(cv2.imread(img0_path), cv2.COLOR_BGR2RGB)
            if os.path.exists(img0_path)
            else np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
        )
        positions, colors = _init_object_from_depth(
            depth_path=depth_path0,
            intrinsics=intrinsics,
            frame_rgb=frame0_rgb,
            object_mask=mask0,
            max_gaussians=cfg.n_gaussians,
            device=device,
        )
    else:
        raise RuntimeError("Provide mesh_path or depth_dir for scene initialization")

    N = positions.shape[0]
    entity_ids = torch.full((N,), ENTITY_OBJECT_BASE, dtype=torch.int32, device=device)
    scene = GaussianScene(positions, colors, entity_ids).to(device)

    init_opacity_raw = math.log(cfg.initial_opacity / (1.0 - cfg.initial_opacity))
    with torch.no_grad():
        scene._opacities_raw.data.fill_(init_opacity_raw)

    print(f"[object_optim] Initialized {N} Gaussians")
    return scene


# --------------------------------------------------------------------------- #
# Optimizer / LR decay
# --------------------------------------------------------------------------- #

def _build_optimizer(
    scene: GaussianScene,
    obj_pose_params: ObjectPoseParams,
    cfg: ObjectOptimConfig,
) -> torch.optim.Adam:
    s = cfg.lr_scale
    return torch.optim.Adam([
        {'params': [scene._positions],         'lr': cfg.lr_positions * s, 'name': 'positions'},
        {'params': [scene._rotations],         'lr': cfg.lr_rotations * s, 'name': 'rotations'},
        {'params': [scene._log_scales],        'lr': cfg.lr_scales    * s, 'name': 'scales'},
        {'params': [scene._opacities_raw],     'lr': cfg.lr_opacities * s, 'name': 'opacities'},
        {'params': [scene._sh_dc],             'lr': cfg.lr_sh_dc     * s, 'name': 'sh_dc'},
        {'params': [scene._sh_rest],           'lr': cfg.lr_sh_rest   * s, 'name': 'sh_rest'},
        {'params': [obj_pose_params.rotations_6d], 'lr': cfg.lr_pose  * s, 'name': 'pose_rot'},
        {'params': [obj_pose_params.translations], 'lr': cfg.lr_pose  * s, 'name': 'pose_transl'},
    ], lr=0.0, eps=1e-15)


def _build_pose_optimizer(
    obj_pose_params: ObjectPoseParams,
    cfg: 'ObjectOptimConfig',
) -> torch.optim.Adam:
    """Pose-only optimizer — used for the per-frame pose estimation phase."""
    s = cfg.lr_scale
    return torch.optim.Adam([
        {'params': [obj_pose_params.rotations_6d], 'lr': cfg.lr_pose * s},
        {'params': [obj_pose_params.translations],  'lr': cfg.lr_pose * s},
    ], lr=0.0, eps=1e-15)


def _build_gaussian_optimizer(
    scene: GaussianScene,
    cfg: 'ObjectOptimConfig',
) -> torch.optim.Adam:
    """Gaussian-only optimizer — used for the canonical refinement phase."""
    s = cfg.lr_scale
    return torch.optim.Adam([
        {'params': [scene._positions],     'lr': cfg.lr_positions * s, 'name': 'positions'},
        {'params': [scene._rotations],     'lr': cfg.lr_rotations * s, 'name': 'rotations'},
        {'params': [scene._log_scales],    'lr': cfg.lr_scales    * s, 'name': 'scales'},
        {'params': [scene._opacities_raw], 'lr': cfg.lr_opacities * s, 'name': 'opacities'},
        {'params': [scene._sh_dc],         'lr': cfg.lr_sh_dc     * s, 'name': 'sh_dc'},
        {'params': [scene._sh_rest],       'lr': cfg.lr_sh_rest   * s, 'name': 'sh_rest'},
    ], lr=0.0, eps=1e-15)


def _build_feat_optimizer(
    feat_gaussians: 'FeatureGaussians',
    cfg: 'ObjectOptimConfig',
) -> torch.optim.Adam:
    """Separate optimizer for FeatureGaussians — survives RGB densification."""
    s = cfg.lr_scale
    return torch.optim.Adam([
        {'params': [feat_gaussians._positions],     'lr': cfg.lr_feat_positions * s},
        {'params': [feat_gaussians._rotations],     'lr': cfg.lr_feat_rotations * s},
        {'params': [feat_gaussians._log_scales],    'lr': cfg.lr_feat_scales    * s},
        {'params': [feat_gaussians._opacities_raw], 'lr': cfg.lr_feat_opacities * s},
        {'params': [feat_gaussians._features],      'lr': cfg.lr_feat_features  * s},
    ], lr=0.0, eps=1e-15)


def _compute_feature_loss(
    feat_gaussians: 'FeatureGaussians',
    obj_pose_params: 'ObjectPoseParams',
    frame_t: int,
    target_feat: torch.Tensor,        # (h, w, D) pre-extracted features
    K_feat: torch.Tensor,             # (1, 3, 3) scaled to feature-map resolution
) -> torch.Tensor:
    """
    Per-frame feature alignment loss for FeatureGaussians.

    Projects each feature Gaussian to 2D, samples target DINOv2 features at that
    location, and minimises cosine distance.  Gradients flow through the projected
    positions back to pose (R, t) and canonical positions.
    """
    from v2d.gsplat.lib.rasterizer import project_and_sample_features
    R, t_vec = obj_pose_params.get_transform(frame_t, 0)
    world_pos = feat_gaussians.positions @ R.T + t_vec.unsqueeze(0)
    gauss_feats, target_feats = project_and_sample_features(
        feat_gaussians, world_pos, K_feat, target_feat,
    )
    if gauss_feats.shape[0] == 0:
        return gauss_feats.sum() * 0.0
    return (1.0 - F.cosine_similarity(gauss_feats, target_feats, dim=-1)).mean()


def _apply_lr_decay(optimizer: torch.optim.Adam, step: int, total: int, cfg: ObjectOptimConfig) -> None:
    progress = step / max(total - 1, 1)
    if cfg.lr_decay_schedule == 'cosine':
        decay = cfg.lr_decay_final + 0.5 * (1.0 - cfg.lr_decay_final) * (1.0 + math.cos(math.pi * progress))
    elif cfg.lr_decay_schedule == 'exponential':
        decay = cfg.lr_decay_final ** progress
    else:
        decay = 1.0
    for g in optimizer.param_groups:
        if 'initial_lr' not in g:
            g['initial_lr'] = g['lr']
        g['lr'] = g['initial_lr'] * decay


# --------------------------------------------------------------------------- #
# Per-frame loss
# --------------------------------------------------------------------------- #

def _compute_frame_loss(
    scene: GaussianScene,
    obj_pose_params: ObjectPoseParams,
    frame_t: int,
    rgb: torch.Tensor,                    # (H, W, 3)
    mask: Optional[torch.Tensor],         # (H, W) float {0,1} or None
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int, W: int,
    cfg: ObjectOptimConfig,
    person_mask: Optional[torch.Tensor] = None,  # (H, W) float {0,1} or None
    depth: Optional[torch.Tensor] = None,         # (H, W) metric depth or None
) -> Tuple[torch.Tensor, dict]:
    R, t_vec = obj_pose_params.get_transform(frame_t, 0)
    world_pos = scene.positions @ R.T + t_vec.unsqueeze(0)
    result = render(scene, world_pos, viewmat, K, H, W, sh_degree=cfg.sh_degree)

    if mask is not None:
        # Zero out non-mask pixels so both losses focus on the object region
        m = mask.unsqueeze(-1)   # (H, W, 1)
        rendered_m = result.rgb * m
        target_m   = rgb * m
    else:
        rendered_m = result.rgb
        target_m   = rgb

    rgb_loss  = F.l1_loss(rendered_m, target_m)
    ssim_loss = loss_ssim(rendered_m, target_m)
    total = cfg.weight_rgb * rgb_loss + cfg.weight_ssim * ssim_loss

    terms = {'rgb': float(rgb_loss), 'ssim': float(ssim_loss)}

    if cfg.weight_mask > 0 and mask is not None:
        # Per-pixel weight map:
        #   inside object mask            → 1.0  (object is visibly here)
        #   outside mask, inside person   → mask_occluder_weight (object may be hidden here)
        #   outside both masks            → mask_outside_weight  (object shouldn't be here)
        alpha = result.alpha.clamp(1e-6, 1 - 1e-6)
        per_pixel = F.binary_cross_entropy(alpha, mask, reduction='none')
        w = torch.where(mask > 0.5,
                        torch.ones_like(mask),
                        torch.full_like(mask, cfg.mask_outside_weight))
        if person_mask is not None:
            occluded_outside = (mask < 0.5) & (person_mask > 0.5)
            w = torch.where(occluded_outside,
                            torch.full_like(w, cfg.mask_occluder_weight),
                            w)
        if cfg.mask_focal_gamma > 0:
            # p_t: model's confidence in the correct binary label
            with torch.no_grad():
                p_t = torch.where(mask > 0.5, alpha, 1.0 - alpha)
            focal_w = (1.0 - p_t) ** cfg.mask_focal_gamma
            mask_loss = (focal_w * per_pixel * w).mean()
        else:
            mask_loss = (per_pixel * w).mean()
        total = total + cfg.weight_mask * mask_loss
        terms['mask'] = float(mask_loss)

    if cfg.weight_centroid > 0 and mask is not None:
        # Project pose translation (object center in camera space) to 2D
        t = t_vec.flatten()                                 # ensure [3]
        k = K[0]                                            # (1,3,3) → (3,3)
        z = t[2].clamp(min=1e-6)
        cx_proj = k[0, 0] * t[0] / z + k[0, 2]
        cy_proj = k[1, 1] * t[1] / z + k[1, 2]

        # Mask centroid in pixel space
        ys = torch.arange(H, device=mask.device, dtype=torch.float32)
        xs = torch.arange(W, device=mask.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        mask_sum = mask.sum().clamp(min=1e-6)
        cx_mask = (grid_x * mask).sum() / mask_sum
        cy_mask = (grid_y * mask).sum() / mask_sum

        centroid_loss = ((cx_proj - cx_mask) / W) ** 2 + ((cy_proj - cy_mask) / H) ** 2
        total = total + cfg.weight_centroid * centroid_loss
        terms['centroid'] = float(centroid_loss)

    if cfg.weight_normal > 0 and depth is not None and mask is not None:
        k = K[0]  # (3, 3)
        n_input    = _depth_to_normals(depth,         k)  # (H, W, 3)
        n_rendered = _depth_to_normals(result.depth,  k)  # (H, W, 3)

        # Valid where both normals are non-zero and inside object mask
        valid_input    = (depth > 0) & (mask > 0.5)
        valid_rendered = result.depth > 0
        valid = valid_input & valid_rendered          # (H, W)

        if valid.any():
            cos_sim = (n_input * n_rendered).sum(dim=-1)  # (H, W)
            normal_loss = (1.0 - cos_sim[valid]).mean()
            total = total + cfg.weight_normal * normal_loss
            terms['normal'] = float(normal_loss)

    return total, terms


# --------------------------------------------------------------------------- #
# Debug renders
# --------------------------------------------------------------------------- #

def _draw_pose_axes(
    img: np.ndarray,          # (H, W, 3) BGR uint8 — modified in place
    R: torch.Tensor,          # (3, 3) canonical-to-world rotation
    center_cam: torch.Tensor, # (3,) centroid of Gaussians in camera space
    K: torch.Tensor,          # (1, 3, 3)
    axis_len: float = 0.1,
) -> None:
    """Project and draw X/Y/Z axes centred on the Gaussian centroid."""
    k = K[0].cpu().numpy()
    R_np = R.cpu().numpy()
    c = center_cam.cpu().numpy().flatten()

    def project(pt: np.ndarray) -> Optional[Tuple[int, int]]:
        if pt[2] < 1e-3:
            return None
        u = int(k[0, 0] * pt[0] / pt[2] + k[0, 2])
        v = int(k[1, 1] * pt[1] / pt[2] + k[1, 2])
        return u, v

    # Axis tips: rotate canonical unit vectors to world space, offset from centroid
    tips = [
        c + R_np @ np.array([axis_len, 0.0,      0.0     ]),  # X — red
        c + R_np @ np.array([0.0,      axis_len,  0.0     ]),  # Y — green
        c + R_np @ np.array([0.0,      0.0,       axis_len]),  # Z — blue
    ]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    p0 = project(c)
    if p0 is None:
        return
    for tip, color in zip(tips, colors):
        p1 = project(tip)
        if p1 is not None:
            cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)
    cv2.circle(img, p0, 4, (255, 255, 255), -1)


def _build_feature_panel(
    feat_gaussians: 'FeatureGaussians',
    obj_pose_params: ObjectPoseParams,
    frame_t: int,
    feat_cache: Dict,
    K_feat: torch.Tensor,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Build a (H, W, 3) BGR panel visualizing DINOv2 features and feature Gaussians.

    Background: target DINOv2 feature map reduced to RGB via per-frame PCA.
    Overlaid dots: each feature Gaussian projected to 2D, colored by its learned
    feature vector through the same PCA basis.

    If features are working, dot colors should match the underlying map at their
    projected locations.
    """
    if frame_t not in feat_cache:
        return np.zeros((H, W, 3), dtype=np.uint8)

    target_feat = feat_cache[frame_t]          # (h, w, D)
    h_f, w_f, D = target_feat.shape

    # PCA: fit on target feature-map pixels (h*w, D)
    feat_flat = target_feat.reshape(-1, D).float().detach()
    mean = feat_flat.mean(0, keepdim=True)      # (1, D)
    centered = feat_flat - mean
    try:
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        pca = Vt[:3]                            # (3, D) top-3 principal components
    except Exception:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Project target features → (h*w, 3), normalise per-channel to [0, 1]
    target_3d = (centered @ pca.T).cpu().numpy()           # (h*w, 3)
    lo = target_3d.min(axis=0, keepdims=True)
    hi = target_3d.max(axis=0, keepdims=True)
    scale = np.where(hi > lo, hi - lo, 1.0)
    target_norm = np.clip((target_3d - lo) / scale, 0.0, 1.0).reshape(h_f, w_f, 3)

    feat_bgr = (target_norm[:, :, ::-1] * 255).astype(np.uint8)
    panel = cv2.resize(feat_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    # Project feature Gaussians and overlay colored dots
    R, t_vec = obj_pose_params.get_transform(frame_t, 0)
    world_pos = feat_gaussians.positions @ R.T + t_vec.unsqueeze(0)
    k = K_feat[0]
    proj = world_pos @ k.T
    z = proj[:, 2]
    u = (proj[:, 0] / z.clamp(min=1e-4)).cpu().numpy()
    v = (proj[:, 1] / z.clamp(min=1e-4)).cpu().numpy()
    valid = ((z > 1e-3) & (proj[:, 0] / z.clamp(min=1e-4) >= 0) &
             (proj[:, 0] / z.clamp(min=1e-4) < w_f) &
             (proj[:, 1] / z.clamp(min=1e-4) >= 0) &
             (proj[:, 1] / z.clamp(min=1e-4) < h_f)).cpu().numpy()

    # Color each Gaussian dot by its learned feature (same PCA + normalization)
    gauss_feats = feat_gaussians.features.float().detach()
    gauss_3d = ((gauss_feats - mean) @ pca.T).cpu().numpy()       # (M, 3)
    gauss_norm = np.clip((gauss_3d - lo) / scale, 0.0, 1.0)       # (M, 3)

    for i in range(len(u)):
        if not valid[i]:
            continue
        px = int(u[i] / w_f * W)
        py = int(v[i] / h_f * H)
        if not (0 <= px < W and 0 <= py < H):
            continue
        r, g, b = gauss_norm[i]
        cv2.circle(panel, (px, py), 3, (int(b * 255), int(g * 255), int(r * 255)), -1)

    cv2.putText(panel, 'features', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return panel


def _save_debug_renders(
    scene: GaussianScene,
    obj_pose_params: ObjectPoseParams,
    frame_data: Dict,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int, W: int,
    batch_idx: int,
    output_dir: str,
    cfg: ObjectOptimConfig,
    n_frames: int = 4,
    foundation_poses: Optional[Dict] = None,
    feat_gaussians: Optional['FeatureGaussians'] = None,
    feat_cache: Optional[Dict] = None,
    K_feat: Optional[torch.Tensor] = None,
) -> None:
    """Save rendered-vs-original PNGs for a sample of frames."""
    debug_dir = os.path.join(output_dir, 'debug', f'batch_{batch_idx:06d}')
    os.makedirs(debug_dir, exist_ok=True)

    frame_indices = list(frame_data.keys())
    sample = frame_indices[::max(1, len(frame_indices) // n_frames)][:n_frames]

    with torch.no_grad():
        for t in sample:
            rgb, mask, _ = frame_data[t]

            R, t_vec = obj_pose_params.get_transform(t, 0)
            world_pos = scene.positions @ R.T + t_vec.unsqueeze(0)
            result = render(scene, world_pos, viewmat, K, H, W, sh_degree=cfg.sh_degree)

            center_cam = world_pos.mean(0)  # (3,) centroid in camera space

            rendered_np = (result.rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            original_np = (rgb.cpu().numpy() * 255).astype(np.uint8)

            rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)

            # Draw mask contour on both panels
            if mask is not None:
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(original_bgr, contours, -1, (0, 255, 0), 2)
                cv2.drawContours(rendered_bgr, contours, -1, (0, 255, 0), 2)

            # Foundation pose axes on original panel (same centroid, FP rotation)
            if foundation_poses is not None and t in foundation_poses:
                R_fp, _ = foundation_poses[t]
                fp_world_pos = scene.positions @ R_fp.T + foundation_poses[t][1].unsqueeze(0)
                fp_center = fp_world_pos.mean(0)
                _draw_pose_axes(original_bgr, R_fp, fp_center, K)

            # Optimized pose axes on rendered panel
            _draw_pose_axes(rendered_bgr, R, center_cam, K)

            panels = [original_bgr, rendered_bgr]

            # Optional feature panel
            if feat_gaussians is not None and feat_cache is not None and K_feat is not None:
                panels.append(_build_feature_panel(
                    feat_gaussians, obj_pose_params, t, feat_cache, K_feat, H, W,
                ))

            combined = np.concatenate(panels, axis=1)
            cv2.putText(combined, f'batch {batch_idx}  frame {t}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, f"{t:06d}.png"), combined)


# --------------------------------------------------------------------------- #
# Curriculum + smoothness helpers
# --------------------------------------------------------------------------- #

def _compute_smoothness_loss(
    obj_pose_params: ObjectPoseParams,
    frame_indices: List[int],
) -> torch.Tensor:
    """
    SE(3) smoothness between consecutive frames.

    Translation: L2 between consecutive translation vectors.
    Rotation: Frobenius norm between consecutive rotation matrices.
    Both are averaged over pairs and summed.
    """
    sorted_frames = sorted(frame_indices)
    if len(sorted_frames) < 2:
        return torch.tensor(0.0, device=obj_pose_params.translations.device)

    trans  = obj_pose_params.translations   # [n_frames, n_obj, 3]
    rot6d  = obj_pose_params.rotations_6d   # [n_frames, n_obj, 6]

    t_curr = torch.stack([trans[t, 0] for t in sorted_frames[:-1]])   # [N-1, 3]
    t_next = torch.stack([trans[t, 0] for t in sorted_frames[1:]])
    r_curr = torch.stack([rot6d[t, 0]  for t in sorted_frames[:-1]])  # [N-1, 6]
    r_next = torch.stack([rot6d[t, 0]  for t in sorted_frames[1:]])

    R_curr = rotation_6d_to_matrix(r_curr)   # [N-1, 3, 3]
    R_next = rotation_6d_to_matrix(r_next)

    trans_loss = F.mse_loss(t_curr, t_next)
    rot_loss   = F.mse_loss(R_curr, R_next)
    return trans_loss + rot_loss


# --------------------------------------------------------------------------- #
# Progressive optimization
# --------------------------------------------------------------------------- #

def _plateau(losses: list, patience: int, threshold: float) -> bool:
    """Return True if the loss has plateaued over the last `patience` steps."""
    if len(losses) < patience:
        return False
    improvement = (losses[-patience] - losses[-1]) / (abs(losses[-patience]) + 1e-8)
    return improvement < threshold



def _run_progressive(
    scene: GaussianScene,
    obj_pose_params: ObjectPoseParams,
    frame_data: Dict,
    depth_data: Dict,
    frame_indices: List[int],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int, W: int,
    cfg: 'ObjectOptimConfig',
    output_dir: str,
    foundation_poses: Optional[Dict] = None,
    feat_gaussians: Optional[FeatureGaussians] = None,
    feat_optimizer: Optional[torch.optim.Adam] = None,
    feat_cache: Optional[Dict] = None,
    K_feat: Optional[torch.Tensor] = None,
) -> GaussianScene:
    """
    For each new frame in temporal order:
      - Warm-start pose from previous frame
      - Jointly optimize pose + canonical on current frame + random history frames
      - Progress to next frame when current frame loss plateaus
    """
    sorted_frames  = sorted(frame_indices)
    optimizer      = _build_optimizer(scene, obj_pose_params, cfg)
    pos_grad_accum = torch.zeros(scene.num_gaussians, device=cfg.device)
    global_step    = 0

    print(f"[object_optim] Progressive: {len(sorted_frames)} frames  "
          f"min={cfg.progressive_min_steps} max={cfg.progressive_max_steps} "
          f"patience={cfg.progressive_patience} threshold={cfg.progressive_plateau_threshold}\n")

    for frame_idx, t in enumerate(sorted_frames):
        # Warm-start pose from previous frame
        if frame_idx > 0:
            prev_t = sorted_frames[frame_idx - 1]
            with torch.no_grad():
                obj_pose_params.rotations_6d[t, 0] = obj_pose_params.rotations_6d[prev_t, 0].clone()
                obj_pose_params.translations[t, 0]  = obj_pose_params.translations[prev_t, 0].clone()

        prev_frames  = sorted_frames[:frame_idx]
        pos_grad_accum = torch.zeros(scene.num_gaussians, device=cfg.device)  # reset per frame
        curr_losses: list = []
        ema = None
        curr_loss = torch.tensor(0.0, device=cfg.device)
        step = 0

        for step in range(cfg.progressive_max_steps):
            optimizer.zero_grad()
            if feat_optimizer is not None:
                feat_optimizer.zero_grad()

            # Current frame — always included; track its loss for plateau detection
            rgb, mask, person_mask = frame_data[t]
            curr_loss, _ = _compute_frame_loss(
                scene, obj_pose_params, t, rgb, mask, viewmat, K, H, W, cfg,
                person_mask=person_mask, depth=depth_data.get(t),
            )
            batch_loss = curr_loss
            n_total = 1

            # Feature alignment for current frame
            if (feat_gaussians is not None and cfg.weight_feature > 0
                    and feat_cache and t in feat_cache):
                feat_loss = _compute_feature_loss(
                    feat_gaussians, obj_pose_params, t, feat_cache[t], K_feat,
                )
                batch_loss = batch_loss + cfg.weight_feature * feat_loss

            # Random sample from history frames
            if prev_frames:
                hist = random.sample(prev_frames, min(cfg.batch_size - 1, len(prev_frames)))
                for ht in hist:
                    rgb_h, mask_h, pm_h = frame_data[ht]
                    h_loss, _ = _compute_frame_loss(
                        scene, obj_pose_params, ht, rgb_h, mask_h, viewmat, K, H, W, cfg,
                        person_mask=pm_h, depth=depth_data.get(ht),
                    )
                    batch_loss = batch_loss + h_loss
                    n_total += 1
                    if (feat_gaussians is not None and cfg.weight_feature > 0
                            and feat_cache and ht in feat_cache):
                        h_feat_loss = _compute_feature_loss(
                            feat_gaussians, obj_pose_params, ht, feat_cache[ht], K_feat,
                        )
                        batch_loss = batch_loss + cfg.weight_feature * h_feat_loss

            batch_loss = batch_loss / n_total

            if cfg.weight_anchor > 0 and scene._anchor_positions is not None:
                batch_loss = batch_loss + cfg.weight_anchor * loss_anchor(
                    scene.positions, scene._anchor_positions
                )

            batch_loss.backward()

            if scene._positions.grad is not None:
                pos_grad_accum += scene._positions.grad.norm(dim=-1).detach()

            optimizer.step()
            if feat_optimizer is not None:
                feat_optimizer.step()
            global_step += 1

            # Densification
            if global_step % cfg.densify_every == 0:
                obj_mask = scene.object_mask(0)
                extent = float(scene.positions[obj_mask].norm(dim=-1).max()) if obj_mask.any() else 10.0
                scene = densify_and_prune(
                    scene, pos_grad_accum,
                    grad_threshold=cfg.grad_threshold,
                    prune_opacity_threshold=cfg.prune_opacity_threshold,
                    max_scene_extent=extent,
                    max_gaussians=cfg.max_gaussians,
                    max_scale_factor=cfg.max_scale_factor,
                )
                pos_grad_accum = torch.zeros(scene.num_gaussians, device=cfg.device)
                optimizer = _build_optimizer(scene, obj_pose_params, cfg)

            # EMA-smooth current frame loss and check plateau
            v = curr_loss.item()
            ema = v if ema is None else cfg.progressive_ema_alpha * v + (1 - cfg.progressive_ema_alpha) * ema
            curr_losses.append(ema)

            if step >= cfg.progressive_min_steps and _plateau(
                curr_losses, cfg.progressive_patience, cfg.progressive_plateau_threshold
            ):
                break

        print(f"  frame {frame_idx+1:>4}/{len(sorted_frames)}  t={t}  "
              f"curr_loss={curr_loss.item():.4f}  steps={step+1}  N={scene.num_gaussians}")

        first  = sorted_frames[0]
        mid    = sorted_frames[frame_idx // 2]
        _save_debug_renders(
            scene, obj_pose_params, {f: frame_data[f] for f in dict.fromkeys([first, mid, t])},
            viewmat, K, H, W, frame_idx + 1, output_dir, cfg,
            foundation_poses=foundation_poses,
            feat_gaussians=feat_gaussians,
            feat_cache=feat_cache,
            K_feat=K_feat,
        )

    return scene


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def gsplat_optimize_object(
    images_dir: str,
    masks_dir: str,
    intrinsics_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    mesh_path: Optional[str] = None,
    depth_dir: Optional[str] = None,
    poses_dir: Optional[str] = None,
    person_masks_dir: Optional[str] = None,
    frame_step: int = 1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    device = cfg.device

    import shutil
    debug_dir = os.path.join(output_dir, 'debug')
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)

    # ------------------------------------------------------------------ #
    # Frames + intrinsics
    # ------------------------------------------------------------------ #
    frame_indices = _find_frame_indices(images_dir, frame_step)
    if not frame_indices:
        raise RuntimeError(f"No frames found in {images_dir}")
    n_frames = max(frame_indices) + 1
    print(f"[object_optim] {len(frame_indices)} frames  device={device}")

    intrinsics_full = CameraIntrinsics.load(intrinsics_path)
    s = cfg.train_scale
    intrinsics_train = CameraIntrinsics(
        fx=intrinsics_full.fx * s, fy=intrinsics_full.fy * s,
        cx=intrinsics_full.cx * s, cy=intrinsics_full.cy * s,
        width=int(intrinsics_full.width * s), height=int(intrinsics_full.height * s),
    )
    H, W = intrinsics_train.height, intrinsics_train.width
    viewmat = build_viewmat(device)
    K = build_K(intrinsics_train, device)

    # ------------------------------------------------------------------ #
    # Frame data + scene + poses
    # ------------------------------------------------------------------ #
    print("[object_optim] Loading frames…")
    frame_data = load_frame_data(images_dir, masks_dir, frame_indices, s, device,
                                 person_masks_dir=person_masks_dir)

    depth_data: Dict[int, torch.Tensor] = {}
    if cfg.weight_normal > 0 and depth_dir is not None:
        print("[object_optim] Loading depth frames for normal loss…")
        depth_data = load_depth_data(depth_dir, frame_indices, s, device)

    scene = init_scene(mesh_path, images_dir, depth_dir, masks_dir,
                       intrinsics_full, frame_indices, poses_dir, cfg).to(device)

    obj_pose_params = ObjectPoseParams(n_frames, num_objects=1, device=device)
    init_poses(obj_pose_params, poses_dir, frame_indices)

    # Snapshot foundation poses before any optimization
    with torch.no_grad():
        foundation_poses = {
            t: (obj_pose_params.get_transform(t, 0)[0].clone(),
                obj_pose_params.get_transform(t, 0)[1].clone())
            for t in frame_indices
        }

    # ------------------------------------------------------------------ #
    # Feature Gaussians setup (optional)
    # ------------------------------------------------------------------ #
    feat_gaussians: Optional[FeatureGaussians] = None
    feat_optimizer: Optional[torch.optim.Adam] = None
    feat_cache: Dict[int, torch.Tensor] = {}
    K_feat: Optional[torch.Tensor] = None

    if cfg.feature_encoder:
        from v2d.gsplat.lib.feature_extractor import FeatureExtractor
        print(f"[object_optim] Initializing feature extractor ({cfg.feature_encoder})…")
        feature_extractor = FeatureExtractor(
            encoder=cfg.feature_encoder,
            proj_dim=cfg.feature_proj_dim,
        ).to(device)
        D_feat = feature_extractor.feature_dim

        print(f"[object_optim] Extracting features for {len(frame_indices)} frames…")
        for t in frame_indices:
            rgb_t, _, _ = frame_data[t]
            feat_cache[t] = feature_extractor.extract(rgb_t)  # (h, w, D)

        h_feat, w_feat = next(iter(feat_cache.values())).shape[:2]

        # K scaled to feature-map resolution
        K_feat = K.clone()
        K_feat[0, 0] = K[0, 0] * (w_feat / W)
        K_feat[0, 1] = K[0, 1] * (h_feat / H)

        # Initialize feature Gaussians from object Gaussian positions
        obj_positions = scene.positions[scene.object_mask(0)].detach().clone()
        if obj_positions.shape[0] > cfg.feature_n_gaussians:
            idx = torch.randperm(obj_positions.shape[0], device=device)[:cfg.feature_n_gaussians]
            obj_positions = obj_positions[idx]
        feat_gaussians = FeatureGaussians(obj_positions, D_feat).to(device)

        # Warm-start feature vectors from frame 0: project positions through the
        # initial (FP) pose, sample DINOv2 features at those 2D locations.
        # This gives semantically meaningful starting features so the cosine loss
        # is informative from step 1, rather than spending iterations climbing out
        # of the random-noise basin.
        if frame_indices and frame_indices[0] in feat_cache:
            from v2d.gsplat.lib.rasterizer import project_and_sample_features
            t0 = frame_indices[0]
            with torch.no_grad():
                R0, t0_vec = obj_pose_params.get_transform(t0, 0)
                world_pos0 = feat_gaussians.positions @ R0.T + t0_vec.unsqueeze(0)
                _, sampled0 = project_and_sample_features(
                    feat_gaussians, world_pos0, K_feat, feat_cache[t0],
                )
                if sampled0.shape[0] == feat_gaussians.num_gaussians:
                    feat_gaussians._features.data.copy_(sampled0)
                elif sampled0.shape[0] > 0:
                    # Some Gaussians projected off-frame — fill in-frame ones only
                    k = K_feat[0]
                    wpos = world_pos0
                    proj = wpos @ k.T
                    z = proj[:, 2]
                    u = proj[:, 0] / z.clamp(min=1e-4)
                    v = proj[:, 1] / z.clamp(min=1e-4)
                    h_f, w_f = feat_cache[t0].shape[:2]
                    valid = (z > 1e-3) & (u >= 0) & (u < w_f) & (v >= 0) & (v < h_f)
                    feat_gaussians._features.data[valid] = sampled0
            print(f"[object_optim] Feature Gaussians warm-started from frame {t0}")

        feat_optimizer = _build_feat_optimizer(feat_gaussians, cfg)
        print(f"[object_optim] Feature Gaussians: {feat_gaussians.num_gaussians} × {D_feat}")

    # ------------------------------------------------------------------ #
    # Optimization loop
    # ------------------------------------------------------------------ #
    if cfg.progressive:
        scene = _run_progressive(
            scene, obj_pose_params, frame_data, depth_data,
            frame_indices, viewmat, K, H, W, cfg, output_dir,
            foundation_poses=foundation_poses,
            feat_gaussians=feat_gaussians,
            feat_optimizer=feat_optimizer,
            feat_cache=feat_cache,
            K_feat=K_feat,
        )
        print("\n[object_optim] Saving outputs…")
        save_gaussians_ply(scene, os.path.join(output_dir, 'gaussians.ply'))
        _save_poses(obj_pose_params, frame_indices, output_dir)
        print(f"[object_optim] Done.  Outputs at: {output_dir}")
        return

    optimizer = _build_optimizer(scene, obj_pose_params, cfg)
    pos_grad_accum  = torch.zeros(scene.num_gaussians, device=device)

    print(f"[object_optim] Training for {cfg.n_batches} batches…\n")

    for i in range(cfg.n_batches):
        batch = random.sample(list(frame_data.keys()), min(cfg.batch_size, len(frame_data)))

        optimizer.zero_grad()
        if feat_optimizer is not None:
            feat_optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        batch_terms: dict = {}

        for t in batch:
            rgb, mask, person_mask = frame_data[t]
            loss, terms = _compute_frame_loss(
                scene, obj_pose_params, t, rgb, mask, viewmat, K, H, W, cfg,
                person_mask=person_mask,
                depth=depth_data.get(t),
            )
            batch_loss = batch_loss + loss / len(batch)
            for k, v in terms.items():
                batch_terms[k] = batch_terms.get(k, 0.0) + v / len(batch)

            if (feat_gaussians is not None and cfg.weight_feature > 0 and t in feat_cache):
                f_loss = _compute_feature_loss(
                    feat_gaussians, obj_pose_params, t, feat_cache[t], K_feat,
                )
                batch_loss = batch_loss + cfg.weight_feature * f_loss / len(batch)
                batch_terms['feat'] = batch_terms.get('feat', 0.0) + float(f_loss) / len(batch)

        # Anchor loss: pull canonical positions toward mesh-init reference.
        # Applied once per step (not per frame) since it's canonical-only.
        if cfg.weight_anchor > 0 and scene._anchor_positions is not None:
            anchor_loss = loss_anchor(scene.positions, scene._anchor_positions)
            batch_loss = batch_loss + cfg.weight_anchor * anchor_loss
            batch_terms['anchor'] = float(anchor_loss)

        # Pose smoothness: applied over ALL frames (not just active curriculum frames)
        # so that bad-frame poses are pulled toward good neighbors even before they
        # contribute to the photometric loss.
        if cfg.weight_smoothness > 0:
            smoothness_loss = _compute_smoothness_loss(obj_pose_params, frame_indices)
            batch_loss = batch_loss + cfg.weight_smoothness * smoothness_loss
            batch_terms['smooth'] = float(smoothness_loss)

        batch_loss.backward()

        if scene._positions.grad is not None:
            pos_grad_accum += scene._positions.grad.norm(dim=-1).detach()

        optimizer.step()
        _apply_lr_decay(optimizer, i, cfg.n_batches, cfg)
        if feat_optimizer is not None:
            feat_optimizer.step()
            _apply_lr_decay(feat_optimizer, i, cfg.n_batches, cfg)

        # Opacity reset
        if cfg.reset_opacity_every > 0 and (i + 1) % cfg.reset_opacity_every == 0:
            with torch.no_grad():
                scene._opacities_raw.data.fill_(-3.0)

        # Densification
        if (i + 1) % cfg.densify_every == 0 and i < cfg.n_batches - 1:
            obj_mask = scene.object_mask(0)
            extent = float(scene.positions[obj_mask].norm(dim=-1).max()) if obj_mask.any() else 10.0
            scene = densify_and_prune(
                scene, pos_grad_accum,
                grad_threshold=cfg.grad_threshold,
                prune_opacity_threshold=cfg.prune_opacity_threshold,
                max_scene_extent=extent,
                max_gaussians=cfg.max_gaussians,
                max_scale_factor=cfg.max_scale_factor,
            )
            pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)
            optimizer = _build_optimizer(scene, obj_pose_params, cfg)

        # Logging + debug renders
        if (i + 1) % cfg.render_interval == 0 or i == 0:
            terms_str = '  '.join(f"{k}={v:.4f}" for k, v in batch_terms.items())
            print(f"  batch {i+1:>5}/{cfg.n_batches}  loss={batch_loss.item():.4f}  "
                  f"{terms_str}  N={scene.num_gaussians}")
            _save_debug_renders(
                scene, obj_pose_params, frame_data,
                viewmat, K, H, W, i + 1, output_dir, cfg,
                feat_gaussians=feat_gaussians,
                feat_cache=feat_cache,
                K_feat=K_feat,
            )

    # ------------------------------------------------------------------ #
    # Save final outputs
    # ------------------------------------------------------------------ #
    print("\n[object_optim] Saving outputs…")
    save_gaussians_ply(scene, os.path.join(output_dir, 'gaussians.ply'))
    _save_poses(obj_pose_params, frame_indices, output_dir)
    print(f"[object_optim] Done.  Outputs at: {output_dir}")


# --------------------------------------------------------------------------- #
# Pose output
# --------------------------------------------------------------------------- #

def _save_poses(
    obj_pose_params: ObjectPoseParams,
    frame_indices: List[int],
    output_dir: str,
) -> None:
    import json
    from scipy.spatial.transform import Rotation as SciRot

    poses_dir = os.path.join(output_dir, 'poses')
    os.makedirs(poses_dir, exist_ok=True)

    with torch.no_grad():
        for t in frame_indices:
            R, t_vec = obj_pose_params.get_transform(t, 0)
            R_np = R.cpu().numpy().astype(np.float64)
            t_np = t_vec.cpu().numpy().astype(np.float64)
            q = SciRot.from_matrix(R_np).as_quat()   # [x,y,z,w]
            record = {
                'rotation':    [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
                'translation': t_np.tolist(),
                'scale':       [1.0, 1.0, 1.0],
            }
            with open(os.path.join(poses_dir, f"{t:06d}.json"), 'w') as f:
                json.dump(record, f, indent=2)

    print(f"[object_optim] Saved {len(frame_indices)} poses to {poses_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rigid-object Gaussian optimization')
    parser.add_argument('--images_dir',      required=True)
    parser.add_argument('--masks_dir',       required=True)
    parser.add_argument('--intrinsics_path', required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--config_path',     default=None)
    parser.add_argument('--mesh_path',       default=None)
    parser.add_argument('--depth_dir',       default=None)
    parser.add_argument('--poses_dir',        default=None)
    parser.add_argument('--person_masks_dir', default=None)
    parser.add_argument('--frame_step',       type=int, default=1)
    args = parser.parse_args()

    gsplat_optimize_object(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        intrinsics_path=args.intrinsics_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
        mesh_path=args.mesh_path,
        depth_dir=args.depth_dir,
        poses_dir=args.poses_dir,
        person_masks_dir=args.person_masks_dir,
        frame_step=args.frame_step,
    )
