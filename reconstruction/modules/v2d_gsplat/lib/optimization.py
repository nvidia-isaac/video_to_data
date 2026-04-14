"""
Alternating canonical / pose optimization for 4D Gaussian Splatting.

Each cycle has two sub-phases:
  Canonical — freeze body/object poses, optimize Gaussian geometry + densify.
  Pose      — freeze Gaussian geometry, optimize body LBS pose + object SE(3).

A final joint refinement phase runs at 0.1× LR after all cycles completes.
"""

import math
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
from v2d.gsplat.lib.deformation import SmplDeformer, BodyPoseParams, ObjectPoseParams, ExposureParams, apply_lbs, rotation_6d_to_matrix
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
    # Per-object global scale correction LR.  Lower than pose LR so scale adapts
    # slowly (avoids depth-scale ambiguity early in training).  0 = frozen.
    lr_obj_scale: float = 1e-4
    # L2 regularization on log_scale — penalises deviating from scale=1.  Keeps
    # scale corrections small unless the rendering loss strongly demands otherwise.
    weight_obj_scale_reg: float = 0.1
    lr_exposure: float = 1e-2   # per-frame log-exposure learning rate
    # L2 penalty on log_exposure — keeps values near 0 (neutral), prevents
    # exposure from absorbing real scene colour variation.
    weight_exposure_reg: float = 0.1
    # ---- LR decay -----------------------------------------------------------
    # Schedule applied uniformly across all cycles + refinement.
    # 'none'        — constant LR (no decay).
    # 'cosine'      — cosine annealing: 1.0 → lr_decay_final over total iters.
    # 'exponential' — exponential: base_lr × lr_decay_final^(t/T).
    lr_decay_schedule: str = 'cosine'
    lr_decay_final: float = 0.1   # final LR as a fraction of its initial value
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
    # Hard-negative mining: sample frames proportional to loss^beta instead of
    # uniformly. beta=0 disables (uniform); beta=1 linear; beta=2 quadratic.
    # eps is the uniform floor — each frame retains at least eps/N probability
    # so no frame is ever completely starved of gradient updates.
    hard_mining_beta: float = 0.0
    hard_mining_eps: float = 0.1
    # Frame sampling strategy across all optimization phases.
    #   'uniform'           — random uniform (baseline).
    #   'hard_negative'     — sample proportional to per-frame loss^beta (use with
    #                         hard_mining_beta > 0; beta=0 falls back to uniform).
    #   'config_diversity'  — sample to maximize spread in human/object pose space;
    #                         under-represented configurations get higher probability.
    #                         Complementary to hard-negative: loss-independent, more
    #                         stable, leverages strong pose priors.
    frame_sampling: str = 'hard_negative'
    # Temperature for config-diversity sampling.  Higher → more uniform;
    # lower → greedier (always picks the most isolated configurations).
    config_diversity_temperature: float = 1.0
    # Per-frame object confidence gating.
    # Before optimization starts, the initial FP pose is rendered for each frame
    # and its IoU with the SAM2 mask is computed.  Frames with IoU × √coverage
    # below min_obj_confidence have their object entity-mask loss zeroed so
    # poorly-tracked / occluded frames do not corrupt canonical shape learning.
    # Those frames' poses are also SLERP-interpolated from neighboring
    # high-confidence frames so the optimization starts from a sensible pose.
    # 0.0 = disabled (all frames weighted equally).
    min_obj_confidence: float = 0.1
    # Anchor object pose parameters toward their SLERP-filled initial values,
    # weighted by (1 - confidence).  Prevents low-confidence frame poses from
    # drifting away from the SLERP interpolation during optimization while
    # letting well-tracked frames refine freely.
    # 0.0 = disabled; typical useful range: 0.5–2.0.
    weight_obj_slerp_anchor: float = 1.0
    device: str = 'cuda'


class HardNegativeSampler:
    """
    Per-frame EMA loss tracker for hard-negative mining.

    Frames are sampled proportional to loss^beta (power-law weighting).
    A uniform floor of eps prevents any frame from being completely starved.

    beta=0  → uniform random (same as no hard mining).
    beta=1  → linear proportional to current loss estimate.
    beta=2  → quadratic — harder frames get even more focus.
    eps     → fraction of probability mass kept uniform across all frames.
    """

    _EMA_ALPHA = 0.05  # smoothing factor — adapts slowly so estimates are stable

    def __init__(
        self,
        frame_indices: List[int],
        batch_size: int,
        beta: float = 1.0,
        eps: float = 0.1,
    ):
        self.frame_indices = list(frame_indices)
        self.batch_size = min(batch_size, len(frame_indices))
        self.beta = beta
        self.eps = eps
        self._loss_ema = np.ones(len(self.frame_indices), dtype=np.float64)
        self._idx_map = {t: i for i, t in enumerate(self.frame_indices)}

    def sample(self) -> List[int]:
        if len(self.frame_indices) <= self.batch_size:
            return list(self.frame_indices)
        if self.beta == 0.0:
            return random.sample(self.frame_indices, self.batch_size)
        hard = self._loss_ema ** self.beta
        hard /= hard.sum()
        uniform = np.ones(len(self.frame_indices)) / len(self.frame_indices)
        probs = (1.0 - self.eps) * hard + self.eps * uniform
        probs /= probs.sum()
        chosen = np.random.choice(
            len(self.frame_indices), size=self.batch_size, replace=False, p=probs
        )
        return [self.frame_indices[i] for i in chosen]

    def update(self, frame_losses: Dict[int, float]):
        """Update EMA loss estimates from the latest batch."""
        a = self._EMA_ALPHA
        for t, v in frame_losses.items():
            if t in self._idx_map:
                i = self._idx_map[t]
                self._loss_ema[i] = (1.0 - a) * self._loss_ema[i] + a * v

    def log_stats(self) -> str:
        losses = self._loss_ema
        return (f"hard_mining: min={losses.min():.4f} mean={losses.mean():.4f} "
                f"max={losses.max():.4f}")


class ConfigDiversitySampler:
    """
    Sample frames to maximize coverage of the human/object configuration space.

    Each frame is represented by a feature vector built from the current SMPL body
    pose (global_orient, body_pose, transl) and per-object SE(3) transforms.
    Frames are sampled proportional to their *isolation* in this space — frames
    whose configuration is far from their neighbours are sampled more often,
    ensuring diverse body/object poses receive proportional training.

    A recency term (inverse of visit-count EMA) prevents any single pose cluster
    from being permanently ignored, even if it's dense.

    Because pose params change slowly relative to the loss, the pairwise distance
    matrix is recomputed every `recompute_every` samples rather than every call.

    update() accepts the same frame_losses dict as HardNegativeSampler (for
    interface compatibility) but only uses it to track visit counts.
    """

    _EMA_DECAY = 0.995
    _RECENCY_WEIGHT = 0.4  # blend between isolation (0) and recency (1)

    def __init__(
        self,
        frame_indices: List[int],
        batch_size: int,
        body_pose_params: Optional['BodyPoseParams'],
        obj_pose_params: Optional['ObjectPoseParams'],
        temperature: float = 1.0,
        recompute_every: int = 50,
    ):
        self.frame_indices = list(frame_indices)
        self.batch_size = min(batch_size, len(frame_indices))
        self.body_pose_params = body_pose_params
        self.obj_pose_params = obj_pose_params
        self.temperature = temperature
        self.recompute_every = recompute_every
        self._n = len(self.frame_indices)
        self._idx_map = {t: i for i, t in enumerate(self.frame_indices)}
        self._visit_ema = np.ones(self._n, dtype=np.float64)
        self._cached_isolation: Optional[np.ndarray] = None
        self._cache_age: int = 0

    def _compute_isolation(self) -> np.ndarray:
        """
        Build normalized pose features, compute pairwise L2 distances, and
        return per-frame isolation scores (mean kNN distance, normalized to sum=1).
        """
        idx = torch.tensor(self.frame_indices, dtype=torch.long)
        parts: List[torch.Tensor] = []

        with torch.no_grad():
            if self.body_pose_params is not None:
                parts.append(self.body_pose_params.global_orient[idx].cpu().float())   # (N, 6)
                parts.append(self.body_pose_params.body_pose[idx].cpu().float())        # (N, J*3)
                parts.append(self.body_pose_params.transl[idx].cpu().float())           # (N, 3)
            if self.obj_pose_params is not None:
                r = self.obj_pose_params.rotations_6d[idx].cpu().float().view(self._n, -1)
                t = self.obj_pose_params.translations[idx].cpu().float().view(self._n, -1)
                parts.extend([r, t])

        if not parts:
            return np.ones(self._n, dtype=np.float64) / self._n

        feats = torch.cat(parts, dim=-1).numpy()   # (N, D)
        std = feats.std(0).clip(min=1e-6)
        feats = (feats - feats.mean(0)) / std      # z-score normalise per dim

        dists = torch.cdist(
            torch.from_numpy(feats), torch.from_numpy(feats)
        ).numpy()                                   # (N, N)

        k = min(5, self._n - 1)
        knn = np.sort(dists, axis=1)[:, 1:k + 1]  # skip self (dist=0)
        isolation = knn.mean(axis=1)               # (N,)
        total = isolation.sum()
        return isolation / total if total > 0.0 else np.ones(self._n) / self._n

    def sample(self) -> List[int]:
        if self._n <= self.batch_size:
            return list(self.frame_indices)

        if self._cached_isolation is None or self._cache_age >= self.recompute_every:
            self._cached_isolation = self._compute_isolation()
            self._cache_age = 0
        self._cache_age += 1

        recency = 1.0 / (self._visit_ema + 1e-6)
        recency /= recency.sum()

        w = self._RECENCY_WEIGHT
        combined = (1.0 - w) * self._cached_isolation + w * recency

        # Temperature-scaled softmax sampling (temperature > 1 → more uniform)
        logits = combined / max(self.temperature, 1e-12)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        chosen = np.random.choice(self._n, size=self.batch_size, replace=False, p=probs)
        return [self.frame_indices[i] for i in chosen]

    def update(self, frame_losses: Dict[int, float]):
        """Update visit count EMA (frame_losses dict is accepted for interface compatibility)."""
        self._visit_ema *= self._EMA_DECAY
        for t in frame_losses:
            if t in self._idx_map:
                self._visit_ema[self._idx_map[t]] += 1.0 - self._EMA_DECAY

    def log_stats(self) -> str:
        v = self._visit_ema
        return (f"config_diversity: visits min={v.min():.3f} "
                f"mean={v.mean():.3f} max={v.max():.3f}")


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
        if scene.skinning_weights is not None:
            # Full LBS with learned skinning weights.
            # Use frame_matrix() to pass the rotation matrix directly to
            # get_joint_transforms, bypassing batch_rodrigues for global_orient
            # and keeping the gradient path free of the θ=π discontinuity.
            go_R, bp, betas, transl = body_pose_params.frame_matrix(frame_t)
            A = smpl_deformer.get_joint_transforms(None, bp, betas, transl, global_orient_R=go_R)  # (1, J, 4, 4)
            A = A.squeeze(0)  # (J, 4, 4)
            canonical_body = scene.positions[body_mask]  # (N_body, 3)
            sw = scene.skinning_weights  # (N_body, J) softmax
            world_body = apply_lbs(canonical_body, sw, A, transl.squeeze(0))
        else:
            # Simpler: use SMPL vertex positions directly
            go, bp, betas, transl = body_pose_params.frame(frame_t)
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
            s_obj = torch.exp(obj_pose_params.log_scales[rid])
            canonical_obj = scene.positions[obj_mask]
            world_obj = s_obj * (canonical_obj @ R.T) + t.unsqueeze(0)
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
            {'params': [obj_pose_params.rotations_6d],  'lr': cfg.lr_obj_pose  * s, 'name': 'obj_rot'},
            {'params': [obj_pose_params.translations],  'lr': cfg.lr_obj_pose  * s, 'name': 'obj_t'},
            {'params': [obj_pose_params.log_scales],    'lr': cfg.lr_obj_scale * s, 'name': 'obj_scale'},
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
    _pose_params   = {'go', 'bp', 'betas', 'transl', 'obj_rot', 'obj_t', 'obj_scale'}

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

    # Store base LRs so the decay scheduler can reconstruct the target at any step.
    for pg in param_groups:
        pg['_base_lr'] = pg['lr']

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
    frame_obj_confidence: Optional[torch.Tensor] = None,
    slerp_obj_r6d: Optional[torch.Tensor] = None,
    slerp_obj_t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[int, float]]:
    """Forward pass over batch_frames; return (total_loss, per-term breakdown, per-frame losses)."""
    batch_loss = torch.tensor(0.0, device=device)
    loss_terms_accum: Dict[str, float] = {}
    per_frame_losses: Dict[int, float] = {}

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
                    # Down-weight object mask loss for frames where the initial FP
                    # pose had poor IoU with SAM2 (occluded or mis-tracked frames).
                    # This prevents those frames from corrupting canonical shape.
                    obj_conf_t = (
                        frame_obj_confidence[t].item()
                        if frame_obj_confidence is not None
                        else 1.0
                    )
                    entity_mask_loss = entity_mask_loss + obj_conf_t * loss_mask_asymmetric(entity_alpha, obj_sam2_mask)
                    n_entity_losses += 1
            if n_entity_losses > 0:
                entity_mask_loss = entity_mask_loss / n_entity_losses
                total = total + entity_mask_weight * entity_mask_loss
                terms['entity_mask'] = entity_mask_loss.item()
            else:
                terms['entity_mask'] = 0.0
        else:
            terms['entity_mask'] = 0.0

        per_frame_losses[t] = total.item()
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

    if cfg.weight_obj_anchor > 0.0 and scene._anchor_positions is not None:
        for rid in range(scene.n_objects()):
            obj_mask_a = scene.object_mask(rid)
            if obj_mask_a.any():
                batch_loss = batch_loss + cfg.weight_obj_anchor * loss_anchor(
                    scene._positions[obj_mask_a],
                    scene._anchor_positions[obj_mask_a].to(device),
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

    # SLERP pose anchor: pull low-confidence object pose frames back toward their
    # SLERP-filled initial values.  Weighted by (1 - confidence) so well-tracked
    # frames are free to refine while FP-failure frames stay near the interpolation.
    # Zero-confidence frames are additionally gradient-masked after backward().
    if (cfg.weight_obj_slerp_anchor > 0.0
            and obj_pose_params is not None
            and frame_obj_confidence is not None
            and slerp_obj_r6d is not None
            and slerp_obj_t is not None):
        T_pose = obj_pose_params.rotations_6d.shape[0]
        anchor_w = (1.0 - frame_obj_confidence[:T_pose]).clamp(0.0, 1.0).to(device)  # (T,)
        dr = obj_pose_params.rotations_6d - slerp_obj_r6d.to(device)  # (T, n_obj, 6)
        dt_v = obj_pose_params.translations - slerp_obj_t.to(device)  # (T, n_obj, 3)
        r_err = (dr ** 2).mean(dim=[1, 2])   # (T,)
        t_err = (dt_v ** 2).mean(dim=[1, 2])  # (T,)
        slerp_loss = (anchor_w * (r_err + t_err)).mean()
        batch_loss = batch_loss + cfg.weight_obj_slerp_anchor * slerp_loss
        loss_terms_accum['slerp_anchor'] = loss_terms_accum.get('slerp_anchor', 0.0) + slerp_loss.item()

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

    # Object scale regularisation — L2 on log_scale keeps corrections small.
    if obj_pose_params is not None and cfg.weight_obj_scale_reg > 0.0:
        scale_reg = obj_pose_params.log_scales.pow(2).mean()
        batch_loss = batch_loss + cfg.weight_obj_scale_reg * scale_reg
        loss_terms_accum['obj_scale_reg'] = scale_reg.item()

    return batch_loss, loss_terms_accum, per_frame_losses


# --------------------------------------------------------------------------- #
# Object-confidence utilities
# --------------------------------------------------------------------------- #

def _compute_obj_frame_confidence(
    frame_indices: List[int],
    obj_pose_params: ObjectPoseParams,
    object_oids: List[int],
    _cache_masks: Dict,
    scene: GaussianScene,
    viewmat: torch.Tensor,
    K_tr: torch.Tensor,
    H_tr: int,
    W_tr: int,
    device: str,
    n_frames_total: int,
) -> torch.Tensor:
    """
    Pre-compute per-frame object confidence using the *initial* FP poses.

    For each frame:  confidence = IoU(rendered silhouette, SAM2 mask) × √coverage

    IoU measures how well the initial FP pose aligns with the SAM2 observation.
    Coverage (mask area / frame area) down-weights frames where the object is
    mostly occluded (small SAM2 mask means little gradient signal anyway).

    Returns a (n_frames_total,) float tensor indexed by absolute frame index.
    Frames not in frame_indices keep the default value of 1.0 (no gating).
    """
    confidence = torch.ones(n_frames_total, device=device)
    if not object_oids:
        return confidence

    with torch.no_grad():
        for t in frame_indices:
            frame_confs = []
            for rid, oid in enumerate(object_oids):
                sam2_mask = _cache_masks.get((oid, t))
                if sam2_mask is None:
                    frame_confs.append(0.0)
                    continue

                sam2_area = sam2_mask.float().mean().item()
                if sam2_area < 1e-3:   # object essentially invisible
                    frame_confs.append(0.0)
                    continue

                entity_sel = scene.entity_ids == (ENTITY_OBJECT_BASE + rid)
                if not entity_sel.any():
                    frame_confs.append(0.0)
                    continue

                # Render object silhouette with current (initial FP) pose.
                # Pass body_pose_params=None / smpl_deformer=None so only
                # the object entities are transformed; body stays at canonical.
                world_pos_t = compute_world_positions(
                    scene, None, obj_pose_params, None, t
                )
                alpha = render_entity_silhouette(
                    scene, entity_sel, world_pos_t, viewmat, K_tr, H_tr, W_tr
                )

                rendered_binary = alpha > 0.5
                sam2_binary     = sam2_mask > 0.5
                intersection    = (rendered_binary & sam2_binary).float().sum()
                union           = (rendered_binary | sam2_binary).float().sum()
                iou             = (intersection / (union + 1e-6)).item()

                frame_confs.append(iou * (sam2_area ** 0.5))

            # Use minimum confidence across objects so all objects must be
            # well-tracked for the frame to be treated as high-confidence.
            confidence[t] = float(min(frame_confs)) if frame_confs else 1.0

    n_total = len(frame_indices)
    n_low   = sum(1 for t in frame_indices if confidence[t].item() < 0.1)
    print(f"  [obj confidence] {n_low}/{n_total} frames below 0.10 "
          f"(mean={confidence[frame_indices].mean().item():.3f})")
    return confidence


def _slerp_fill_object_poses(
    obj_pose_params: ObjectPoseParams,
    frame_obj_confidence: torch.Tensor,   # (n_frames_total,) indexed by frame_t
    frame_indices: List[int],
    min_confidence: float = 0.1,
) -> None:
    """
    For each object, replace poses of frames with confidence < min_confidence
    by SLERP interpolation from neighboring high-confidence frames.

    Rotation: SLERP via scipy (smooth geodesic path on SO(3)).
    Translation: linear interpolation.

    Operates in-place on obj_pose_params.rotations_6d / translations.
    Frames outside [first_hc, last_hc] are clamped to the nearest boundary.
    """
    from scipy.spatial.transform import Rotation as SciRot, Slerp as SciSlerp

    with torch.no_grad():
        T, n_obj, _ = obj_pose_params.rotations_6d.shape
        dev = obj_pose_params.rotations_6d.device
        conf_np = frame_obj_confidence.cpu().numpy()

        for oid in range(n_obj):
            hc_idx = np.array(
                [t for t in frame_indices if conf_np[t] >= min_confidence],
                dtype=int,
            )
            if len(hc_idx) < 2:
                print(f"  [slerp] obj {oid}: fewer than 2 high-confidence frames "
                      f"— skipping interpolation")
                continue

            # Build scipy Slerp from high-confidence rotations
            r6d_all  = obj_pose_params.rotations_6d[:, oid].cpu().float()  # (T, 6)
            R_mats   = rotation_6d_to_matrix(r6d_all).numpy()              # (T, 3, 3)
            R_hc     = SciRot.from_matrix(R_mats[hc_idx])                  # n_hc rotations
            slerp_fn = SciSlerp(hc_idx.astype(float), R_hc)

            t_hc_min, t_hc_max = int(hc_idx[0]), int(hc_idx[-1])
            lc_frames = [t for t in frame_indices if conf_np[t] < min_confidence]

            for t in lc_frames:
                if t <= t_hc_min:
                    R_new = torch.from_numpy(R_mats[t_hc_min].astype(np.float32))
                    t_new = obj_pose_params.translations[t_hc_min, oid].clone()
                elif t >= t_hc_max:
                    R_new = torch.from_numpy(R_mats[t_hc_max].astype(np.float32))
                    t_new = obj_pose_params.translations[t_hc_max, oid].clone()
                else:
                    # SLERP rotation
                    R_new = torch.from_numpy(
                        slerp_fn(float(t)).as_matrix().astype(np.float32)
                    )
                    # Linear translation between the two bounding high-conf frames
                    left  = int(hc_idx[hc_idx <= t].max())
                    right = int(hc_idx[hc_idx >= t].min())
                    alpha = (t - left) / (right - left)
                    t_new = ((1.0 - alpha) * obj_pose_params.translations[left,  oid] +
                             alpha          * obj_pose_params.translations[right, oid])

                # Encode back as 6D (first two columns of the rotation matrix)
                r6d_new = torch.cat([R_new[:, 0], R_new[:, 1]])
                obj_pose_params.rotations_6d.data[t, oid] = r6d_new.to(dev)
                obj_pose_params.translations.data[t, oid] = t_new.to(dev)

            print(f"  [slerp] obj {oid}: filled {len(lc_frames)} low-confidence frames "
                  f"from {len(hc_idx)} anchors (threshold={min_confidence:.2f})")


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
        # frame_obj_confidence is assigned after _loss_kwargs — set placeholder here,
        # will be replaced once the confidence tensor is computed below.
        frame_obj_confidence=None,
    )

    if cfg.frame_sampling == 'config_diversity':
        sampler = ConfigDiversitySampler(
            frame_indices, cfg.batch_size,
            body_pose_params=body_pose_params,
            obj_pose_params=obj_pose_params,
            temperature=cfg.config_diversity_temperature,
        )
        print(f"  [optim] Config-diversity sampling enabled "
              f"(temp={cfg.config_diversity_temperature})")
    else:
        sampler = HardNegativeSampler(
            frame_indices, cfg.batch_size,
            beta=cfg.hard_mining_beta,
            eps=cfg.hard_mining_eps,
        )
        if cfg.hard_mining_beta > 0.0:
            print(f"  [optim] Hard-negative mining enabled "
                  f"(beta={cfg.hard_mining_beta}, eps={cfg.hard_mining_eps})")

    # Snapshot initial object poses before any optimization modifies them.
    # Used in checkpoint renders to compare FP-initialised vs optimised object poses.
    if obj_pose_params is not None and scene.n_objects() > 0:
        _init_obj_r6d      = obj_pose_params.rotations_6d.detach().clone()  # (T, n_obj, 6)
        _init_obj_t        = obj_pose_params.translations.detach().clone()   # (T, n_obj, 3)
        _init_obj_logscale = obj_pose_params.log_scales.detach().clone()     # (n_obj,)
    else:
        _init_obj_r6d = _init_obj_t = _init_obj_logscale = None

    # Per-frame object confidence: IoU(initial FP render, SAM2 mask) × √coverage.
    # Computed once before training using the as-loaded FP poses.
    # Frames below min_obj_confidence:
    #   - have their object poses SLERP-interpolated from neighbors (better init)
    #   - have zero weight on the object entity-mask loss (canonical not corrupted)
    if cfg.min_obj_confidence > 0.0 and obj_pose_params is not None and object_oids:
        print(f"  [obj confidence] Computing per-frame object confidence scores...", flush=True)
        frame_obj_confidence = _compute_obj_frame_confidence(
            frame_indices=frame_indices,
            obj_pose_params=obj_pose_params,
            object_oids=object_oids,
            _cache_masks=_cache_masks,
            scene=scene,
            viewmat=viewmat,
            K_tr=K_tr,
            H_tr=H_tr,
            W_tr=W_tr,
            device=device,
            n_frames_total=n_frames_total,
        )
        _slerp_fill_object_poses(
            obj_pose_params=obj_pose_params,
            frame_obj_confidence=frame_obj_confidence,
            frame_indices=frame_indices,
            min_confidence=cfg.min_obj_confidence,
        )
        # Snapshot SLERP-filled poses as anchors BEFORE any optimization.
        # Used to prevent low-confidence frames drifting during training.
        _slerp_obj_r6d = obj_pose_params.rotations_6d.detach().clone()  # (T, n_obj, 6)
        _slerp_obj_t   = obj_pose_params.translations.detach().clone()   # (T, n_obj, 3)
        # Zero out frames below threshold so they contribute nothing to the
        # object entity-mask loss; frames above keep their raw [0,1] score.
        frame_obj_confidence = torch.where(
            frame_obj_confidence >= cfg.min_obj_confidence,
            frame_obj_confidence,
            torch.zeros_like(frame_obj_confidence),
        )
    else:
        frame_obj_confidence = None
        _slerp_obj_r6d = None
        _slerp_obj_t   = None

    # Patch into _loss_kwargs now that the tensor is ready.
    _loss_kwargs['frame_obj_confidence'] = frame_obj_confidence
    _loss_kwargs['slerp_obj_r6d'] = _slerp_obj_r6d
    _loss_kwargs['slerp_obj_t']   = _slerp_obj_t

    # Total gradient steps across all cycles + refinement — used by LR decay.
    _iters_per_cycle = cfg.iterations_canonical_per_cycle + cfg.iterations_pose_per_cycle
    _total_training_iters = cfg.n_cycles * _iters_per_cycle + cfg.iterations_refine

    def _apply_lr_decay(opt: optim.Adam) -> None:
        """Scale all optimizer LRs by the decay factor for the current step."""
        if cfg.lr_decay_schedule == 'none' or cfg.lr_decay_final >= 1.0:
            return
        t = min(total_iter / max(_total_training_iters, 1), 1.0)
        if cfg.lr_decay_schedule == 'cosine':
            factor = cfg.lr_decay_final + (1.0 - cfg.lr_decay_final) * 0.5 * (1.0 + math.cos(math.pi * t))
        else:  # 'exponential'
            factor = cfg.lr_decay_final ** t
        for pg in opt.param_groups:
            pg['lr'] = pg.get('_base_lr', pg['lr']) * factor

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

    def _world_pos_with_obj_override(frame_t, r6d_snap, t_snap, logscale_snap):
        """World positions using body's current pose but overridden object SE(3)+scale."""
        # Pass obj_pose_params=None so object entities stay at canonical; we override below.
        world_pos = compute_world_positions(
            scene, body_pose_params, None, smpl_deformer, frame_t
        )
        for rid in range(scene.n_objects()):
            obj_mask = scene.object_mask(rid)
            if not obj_mask.any():
                continue
            R = rotation_6d_to_matrix(r6d_snap[frame_t, rid].unsqueeze(0)).squeeze(0)
            t = t_snap[frame_t, rid]
            s = torch.exp(logscale_snap[rid])
            canonical_obj = scene.positions[obj_mask]
            world_pos = world_pos.clone()
            world_pos[obj_mask] = s * (canonical_obj @ R.T) + t.unsqueeze(0)
        return world_pos

    def _checkpoint(tag: str, frame_t: int):
        try:
            orig_rgb = _load_frame_rgb(video_path, frame_t)  # (H, W, 3) uint8
            if orig_rgb.shape[0] != H or orig_rgb.shape[1] != W:
                orig_rgb = cv2.resize(orig_rgb, (W, H))
        except Exception:
            orig_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label_h = 40

        def _make_row(panels, labels, col=(255, 255, 255)):
            row = np.zeros((H + label_h, 4 * W, 3), dtype=np.uint8)
            for j, (panel, lbl) in enumerate(zip(panels, labels)):
                row[label_h:, j*W:(j+1)*W] = panel
                cv2.putText(row, lbl, (j*W + 10, 28), font, 0.9, col, 2)
            return row

        with torch.no_grad():
            _wp = compute_world_positions(
                scene, body_pose_params, obj_pose_params, smpl_deformer, frame_t
            )
            rend_orig = _render_np(scene, _wp, viewmat,      K, H, W, cfg.sh_degree, device)
            rend_m30  = _render_np(scene, _wp, _viewmat_m30, K, H, W, cfg.sh_degree, device)
            rend_p30  = _render_np(scene, _wp, _viewmat_p30, K, H, W, cfg.sh_degree, device)

        rows = [_make_row(
            [orig_rgb, rend_orig, rend_m30, rend_p30],
            ['Original', 'Rendered', '-30 deg', '+30 deg'],
        )]

        # Object pose comparison rows — only when object entities are present.
        if _init_obj_r6d is not None:
            with torch.no_grad():
                _wp_fp = _world_pos_with_obj_override(
                    frame_t, _init_obj_r6d, _init_obj_t, _init_obj_logscale
                )
                fp_orig = _render_np(scene, _wp_fp, viewmat,      K, H, W, cfg.sh_degree, device)
                fp_m30  = _render_np(scene, _wp_fp, _viewmat_m30, K, H, W, cfg.sh_degree, device)
                fp_p30  = _render_np(scene, _wp_fp, _viewmat_p30, K, H, W, cfg.sh_degree, device)

            rows.append(_make_row(
                [orig_rgb, fp_orig, fp_m30, fp_p30],
                ['Original', 'FP obj pose', 'FP -30', 'FP +30'],
                col=(255, 200, 50),   # amber — initial FP pose
            ))
            rows.append(_make_row(
                [orig_rgb, rend_orig, rend_m30, rend_p30],
                ['Original', 'Opt obj pose', 'Opt -30', 'Opt +30'],
                col=(100, 255, 100),  # green — current optimised pose
            ))

        out_path = os.path.join(renders_dir, f"{tag}.png")
        try:
            cv2.imwrite(out_path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
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
        _apply_lr_decay(optimizer)  # sync to current position in decay schedule
        pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)
        pos_grad_count = torch.zeros(scene.num_gaussians, device=device)

        for i in range(n_iters):
            batch_frames = sampler.sample()
            compute_entity_mask = (
                cfg.loss_weights.get('entity_mask', 1.0) > 0
                and (i % cfg.entity_mask_interval == 0)
            )
            optimizer.zero_grad()
            batch_loss, terms, frame_losses = _compute_batch_loss(
                scene, batch_frames, compute_entity_mask=compute_entity_mask,
                **_loss_kwargs
            )
            batch_loss.backward()

            # Hard-freeze zero-confidence object pose frames: zero their gradients
            # so Adam doesn't accumulate momentum from bad FP frames.
            # The SLERP anchor loss handles soft regularization for non-zero
            # low-confidence frames; gradient masking handles the completely bad ones.
            if obj_pose_params is not None and frame_obj_confidence is not None:
                T_pose = obj_pose_params.rotations_6d.shape[0]
                bad = (frame_obj_confidence[:T_pose] == 0.0).nonzero(as_tuple=True)[0]
                if len(bad) > 0:
                    if obj_pose_params.rotations_6d.grad is not None:
                        obj_pose_params.rotations_6d.grad[bad] = 0.0
                    if obj_pose_params.translations.grad is not None:
                        obj_pose_params.translations.grad[bad] = 0.0

            sampler.update(frame_losses)

            if do_densify and scene._positions.grad is not None:
                pos_grad_accum += scene._positions.grad.norm(dim=-1).detach()
                pos_grad_count += 1

            optimizer.step()
            nonlocal total_iter
            total_iter += 1
            _apply_lr_decay(optimizer)
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
                _apply_lr_decay(optimizer)
                pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)
                pos_grad_count = torch.zeros(scene.num_gaussians, device=device)
                print(f"  [densify] N={scene.num_gaussians}")

            if do_densify and cfg.reset_opacity_every > 0 and (i + 1) % cfg.reset_opacity_every == 0:
                with torch.no_grad():
                    # sigmoid(-4) ≈ 0.018 — small but nonzero so Gaussians aren't
                    # immediately pruned; they must re-earn opacity from the loss.
                    scene._opacities_raw.data.fill_(-4.0)
                optimizer = _setup_optimizer(scene, body_pose_params, obj_pose_params, cfg, opt_mode, exposure_params)
                _apply_lr_decay(optimizer)
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
    _apply_lr_decay(optimizer)
    print(f"\n--- Refinement ({n_iters} iters, 0.1× LR) ---")

    for i in range(n_iters):
        batch_frames = sampler.sample()
        compute_entity_mask = (
            cfg.loss_weights.get('entity_mask', 1.0) > 0
            and (i % cfg.entity_mask_interval == 0)
        )
        optimizer.zero_grad()
        batch_loss, terms, frame_losses = _compute_batch_loss(
            scene, batch_frames, compute_entity_mask=compute_entity_mask,
            **_loss_kwargs
        )
        batch_loss.backward()
        sampler.update(frame_losses)
        optimizer.step()
        total_iter += 1
        _apply_lr_decay(optimizer)
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
                batch_loss, terms, frame_losses = _compute_batch_loss(
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
