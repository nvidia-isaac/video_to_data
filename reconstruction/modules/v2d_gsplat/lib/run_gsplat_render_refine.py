"""
Two-phase render-and-compare Gaussian object tracking.

Phase 1 — Canonical training with quality-filtered keyframes:
  Score every FP-pose frame by temporal consistency (large pose-velocity jumps
  indicate unreliable FP estimates).  Divide the sequence into n_keyframes
  equal temporal bins and pick the most consistent frame from each bin to
  guarantee diverse viewpoint coverage.  Train the canonical Gaussian with poses
  FIXED — removing the canonical/pose ambiguity that plagues joint optimisation.

Phase 2 — Render-and-compare tracking:
  With a frozen canonical, refine every per-frame pose by minimising the
  render-vs-observed loss (masked L1 + SSIM + silhouette BCE).  Each frame is
  warm-started from the previous refined pose, providing temporal coherence.
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

from v2d.common.datatypes import CameraIntrinsics
from v2d.gsplat.lib.scene import GaussianScene, ENTITY_OBJECT_BASE
from v2d.gsplat.lib.deformation import ObjectPoseParams
from v2d.gsplat.lib.rasterizer import render, build_viewmat, build_K
from v2d.gsplat.lib.losses import loss_ssim, loss_anchor
from v2d.gsplat.lib.densification import densify_and_prune
from v2d.gsplat.lib.extraction import save_gaussians_ply

# Reuse data-loading and initialisation helpers from the joint-optimisation script
from v2d.gsplat.lib.run_gsplat_optimize_object import (
    load_frame_data,
    _find_frame_indices,
    _load_absolute_transform,
    _apply_lr_decay,
    _draw_pose_axes,
    init_scene,
    init_poses,
    _save_poses,
    ObjectOptimConfig as _OC,   # needed only to satisfy init_scene's type hint
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class RenderRefineConfig:
    # ---- Keyframe selection ------------------------------------------------
    n_keyframes: int = 100          # max keyframes for canonical training
    # Frames whose pose-velocity exceeds this multiple of the rolling median
    # are penalised as candidate keyframes.
    temporal_outlier_factor: float = 3.0
    temporal_window: int = 5        # half-width of rolling median window
    keyframe_min_spacing: int = 3   # enforce minimum gap between keyframes

    # ---- Canonical training (Phase 1) -------------------------------------
    n_batches: int = 5000
    batch_size: int = 8
    render_interval: int = 500
    weight_rgb: float = 1.0
    weight_ssim: float = 0.2
    weight_mask: float = 1.0
    mask_outside_weight: float = 0.1
    mask_occluder_weight: float = 0.0    # outside obj mask but inside person mask (may be hidden)
    mask_focal_gamma: float = 2.0
    weight_anchor: float = 0.1
    densify_every: int = 500
    grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.005
    max_gaussians: int = 100_000
    max_scale_factor: float = 0.1
    reset_opacity_every: int = 3000
    lr_decay_schedule: str = 'cosine'
    lr_decay_final: float = 0.1

    # ---- Gaussian learning rates ------------------------------------------
    lr_scale: float = 1.0
    lr_positions: float = 1.6e-4
    lr_rotations: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 1e-3
    lr_sh_rest: float = 1e-4

    # ---- Tracking (Phase 2) -----------------------------------------------
    n_steps_track: int = 200
    track_weight_rgb: float = 1.0
    track_weight_ssim: float = 0.2
    track_weight_mask: float = 1.0
    track_mask_outside_weight: float = 0.1
    track_mask_occluder_weight: float = 0.0  # outside obj mask but inside person mask
    track_mask_focal_gamma: float = 2.0
    lr_pose_canonical: float = 1e-4   # pose LR during canonical training (small corrections only)
    lr_pose_track: float = 1e-3
    track_min_steps: int = 10
    track_plateau_patience: int = 20
    track_plateau_threshold: float = 1e-4
    track_ema_alpha: float = 0.3
    track_debug_every: int = 10     # save debug image every N tracked frames

    # ---- Scene ------------------------------------------------------------
    sh_degree: int = 3
    train_scale: float = 0.5
    n_gaussians: int = 15_000
    initial_opacity: float = 0.5
    device: str = 'cuda'


def load_config(config_path: Optional[str]) -> RenderRefineConfig:
    cfg = RenderRefineConfig()
    if not config_path or not os.path.exists(config_path):
        return cfg
    import yaml
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f"[render_refine] WARNING: unknown config key '{k}' — ignored")
    return cfg


# --------------------------------------------------------------------------- #
# Keyframe selection
# --------------------------------------------------------------------------- #

def _pose_velocity(
    pose_params: ObjectPoseParams,
    sorted_frames: List[int],
) -> np.ndarray:
    """
    SE(3) distance between consecutive FP poses.
    Returns (len(sorted_frames),) float32; velocity[0] = 0.
    """
    vel = np.zeros(len(sorted_frames), dtype=np.float32)
    with torch.no_grad():
        for i in range(1, len(sorted_frames)):
            t0, t1 = sorted_frames[i - 1], sorted_frames[i]
            R0, tr0 = pose_params.get_transform(t0, 0)
            R1, tr1 = pose_params.get_transform(t1, 0)
            vel[i] = float((tr1 - tr0).norm()) + float((R1 - R0).norm())
    return vel


def _temporal_quality(
    vel: np.ndarray,
    window: int,
    outlier_factor: float,
) -> np.ndarray:
    """
    Score each frame by how well its velocity matches local neighbours.
    Returns scores in (0, 1]; low = unexpected velocity spike = bad FP frame.
    """
    n = len(vel)
    scores = np.ones(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        nbrs = np.concatenate([vel[lo:i], vel[i + 1:hi]])
        if len(nbrs) == 0:
            continue
        expected = float(np.median(nbrs)) + 1e-6
        ratio = vel[i] / expected
        scores[i] = 1.0 / (1.0 + max(0.0, ratio - outlier_factor))
    return scores


def _mesh_silhouette(
    verts_cam: np.ndarray,   # (V, 3) already in camera space
    faces: np.ndarray,        # (F, 3) int32
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project mesh triangles into a binary silhouette mask (H, W) uint8."""
    z = verts_cam[:, 2]
    valid_v = z > 1e-4
    safe_z = np.where(valid_v, z, 1.0)
    u = fx * verts_cam[:, 0] / safe_z + cx
    v = fy * verts_cam[:, 1] / safe_z + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)

    # Keep only faces where all three vertices are in front of camera
    front = valid_v[faces]                          # (F, 3) bool
    keep = front.all(axis=1)
    valid_faces = faces[keep]

    # Collect polygon arrays for batch fillPoly
    polygons = []
    for tri in valid_faces:
        pts = uv[tri].round().astype(np.int32)
        # Skip if entirely off-screen
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= W:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= H:
            continue
        polygons.append(pts.reshape(-1, 1, 2))

    canvas = np.zeros((H, W), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(canvas, polygons, 255)
    return canvas


def _compute_iou_scores(
    mesh_path: str,
    poses_dir: str,
    frame_indices: List[int],
    frame_data: Dict[int, Tuple],
    intrinsics_train: CameraIntrinsics,
) -> Dict[int, float]:
    """
    For each frame, render the FP mesh silhouette at the absolute FP pose and
    compute IoU against the SAM2 object mask (excluding person pixels).

    Returns a dict {frame_idx: iou_score}.  Frames whose pose JSON is missing
    receive iou=0.0 so they are deprioritised in keyframe selection.
    """
    try:
        import trimesh as _trimesh
    except ImportError:
        print("[render_refine] trimesh not found — IoU scoring skipped")
        return {}

    H, W = intrinsics_train.height, intrinsics_train.width
    fx, fy = float(intrinsics_train.fx), float(intrinsics_train.fy)
    cx, cy = float(intrinsics_train.cx), float(intrinsics_train.cy)

    mesh = _trimesh.load(mesh_path, force='mesh', process=False)
    verts_obj = np.array(mesh.vertices, dtype=np.float64)   # (V, 3)
    faces     = np.array(mesh.faces,    dtype=np.int32)      # (F, 3)
    print(f"[render_refine] IoU scoring: mesh has {len(verts_obj)} verts, "
          f"{len(faces)} faces — {len(frame_indices)} frames")

    iou_scores: Dict[int, float] = {}
    for t in frame_indices:
        pose_path = os.path.join(poses_dir, f"{t:06d}.json")
        if not os.path.exists(pose_path):
            iou_scores[t] = 0.0
            continue

        R, tr = _load_absolute_transform(pose_path)
        # Transform vertices to camera space: v_cam = R @ v_obj + t
        verts_cam = (verts_obj @ R.T + tr).astype(np.float32)

        sil = _mesh_silhouette(verts_cam, faces, H, W, fx, fy, cx, cy)

        if t not in frame_data:
            iou_scores[t] = 0.0
            continue

        _, obj_mask, person_mask = frame_data[t]
        if obj_mask is None:
            iou_scores[t] = 0.5   # neutral — no mask to compare against
            continue

        target   = obj_mask.cpu().numpy() > 0.5       # (H, W) bool
        rendered = sil > 127                           # (H, W) bool

        if person_mask is not None:
            exclude  = person_mask.cpu().numpy() > 0.5
            target   = target & ~exclude
            rendered = rendered & ~exclude

        intersection = int((target & rendered).sum())
        union        = int((target | rendered).sum())
        iou_scores[t] = intersection / max(union, 1)

    valid = [v for v in iou_scores.values() if v > 0]
    mean_iou = float(np.mean(valid)) if valid else 0.0
    print(f"[render_refine] IoU scores computed — mean={mean_iou:.3f} "
          f"over {len(valid)} frames with pose")
    return iou_scores


def select_keyframes(
    pose_params: ObjectPoseParams,
    frame_indices: List[int],
    cfg: RenderRefineConfig,
    frame_data: Optional[Dict[int, Tuple]] = None,
    mesh_path: Optional[str] = None,
    poses_dir: Optional[str] = None,
    intrinsics_train: Optional[CameraIntrinsics] = None,
) -> List[int]:
    """
    Select up to cfg.n_keyframes frames for canonical training.

    Strategy: divide the sequence into n_keyframes equal temporal bins;
    from each bin pick the frame with the best combined score.

    Score = temporal_consistency × IoU(rendered_mesh_silhouette, SAM2_mask)

    IoU is only included when mesh_path, poses_dir, frame_data, and
    intrinsics_train are all provided.  If any is missing the selection falls
    back to temporal consistency alone.

    A minimum-spacing filter avoids near-duplicate viewpoints.
    """
    sorted_frames = sorted(frame_indices)
    vel = _pose_velocity(pose_params, sorted_frames)
    temporal_scores = _temporal_quality(vel, cfg.temporal_window, cfg.temporal_outlier_factor)

    # Optionally weight by mesh-vs-SAM2 IoU
    if mesh_path and poses_dir and frame_data is not None and intrinsics_train is not None:
        iou_map = _compute_iou_scores(
            mesh_path, poses_dir, sorted_frames, frame_data, intrinsics_train
        )
        if iou_map:
            iou_arr = np.array([iou_map.get(t, 0.0) for t in sorted_frames], dtype=np.float32)
            combined = temporal_scores * iou_arr
        else:
            combined = temporal_scores
    else:
        combined = temporal_scores

    score_map = {t: float(s) for t, s in zip(sorted_frames, combined)}

    n = len(sorted_frames)
    k = min(cfg.n_keyframes, n)

    # Best-per-bin selection for temporal coverage
    binned: List[int] = []
    for b in range(k):
        lo = b * n // k
        hi = (b + 1) * n // k
        candidates = sorted_frames[lo:hi]
        if candidates:
            binned.append(max(candidates, key=lambda t: score_map[t]))

    # Minimum spacing filter
    selected: List[int] = []
    for t in binned:
        if all(abs(t - s) >= cfg.keyframe_min_spacing for s in selected):
            selected.append(t)

    return sorted(selected)


# --------------------------------------------------------------------------- #
# Canonical training (Phase 1) — Gaussian-only optimizer, fixed poses
# --------------------------------------------------------------------------- #

def _canonical_optimizer(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    cfg: RenderRefineConfig,
) -> torch.optim.Adam:
    s = cfg.lr_scale
    return torch.optim.Adam([
        {'params': [scene._positions],          'lr': cfg.lr_positions     * s, 'name': 'positions'},
        {'params': [scene._rotations],          'lr': cfg.lr_rotations     * s, 'name': 'rotations'},
        {'params': [scene._log_scales],         'lr': cfg.lr_scales        * s, 'name': 'scales'},
        {'params': [scene._opacities_raw],      'lr': cfg.lr_opacities     * s, 'name': 'opacities'},
        {'params': [scene._sh_dc],              'lr': cfg.lr_sh_dc         * s, 'name': 'sh_dc'},
        {'params': [scene._sh_rest],            'lr': cfg.lr_sh_rest       * s, 'name': 'sh_rest'},
        {'params': [pose_params.rotations_6d],  'lr': cfg.lr_pose_canonical * s, 'name': 'pose_rot'},
        {'params': [pose_params.translations],   'lr': cfg.lr_pose_canonical * s, 'name': 'pose_transl'},
    ], lr=0.0, eps=1e-15)


def _canonical_frame_loss(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    frame_t: int,
    rgb: torch.Tensor,
    mask: Optional[torch.Tensor],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    cfg: RenderRefineConfig,
    person_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    R, t_vec = pose_params.get_transform(frame_t, 0)
    world_pos = scene.positions @ R.T + t_vec.unsqueeze(0)
    result = render(scene, world_pos, viewmat, K, H, W, sh_degree=cfg.sh_degree)

    m = mask.unsqueeze(-1) if mask is not None else torch.ones_like(rgb[:, :, :1])
    rgb_loss  = F.l1_loss(result.rgb * m, rgb * m)
    ssim_loss = loss_ssim(result.rgb * m, rgb * m)
    total = cfg.weight_rgb * rgb_loss + cfg.weight_ssim * ssim_loss

    if cfg.weight_mask > 0 and mask is not None:
        alpha = result.alpha.clamp(1e-6, 1 - 1e-6)
        per_pixel = F.binary_cross_entropy(alpha, mask, reduction='none')
        w = torch.where(mask > 0.5,
                        torch.ones_like(mask),
                        torch.full_like(mask, cfg.mask_outside_weight))
        if person_mask is not None:
            occluded = (mask < 0.5) & (person_mask > 0.5)
            w = torch.where(occluded, torch.full_like(w, cfg.mask_occluder_weight), w)
        if cfg.mask_focal_gamma > 0:
            with torch.no_grad():
                p_t = torch.where(mask > 0.5, alpha, 1.0 - alpha)
                focal_w = (1.0 - p_t) ** cfg.mask_focal_gamma
            mask_loss = (focal_w * per_pixel * w).mean()
        else:
            mask_loss = (per_pixel * w).mean()
        total = total + cfg.weight_mask * mask_loss

    return total


def train_canonical(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    keyframe_data: Dict[int, Tuple],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    cfg: RenderRefineConfig,
    output_dir: str,
) -> GaussianScene:
    """
    Train canonical Gaussians with poses FIXED on selected keyframes.
    No pose parameters in the optimizer — all gradient goes to shape/appearance.
    """
    device = cfg.device
    keyframes = list(keyframe_data.keys())
    optimizer = _canonical_optimizer(scene, pose_params, cfg)
    pos_grad_accum = torch.zeros(scene.num_gaussians, device=device)

    print(f"[render_refine] Canonical training: {len(keyframes)} keyframes  "
          f"{cfg.n_batches} batches\n")

    for i in range(cfg.n_batches):
        batch = random.sample(keyframes, min(cfg.batch_size, len(keyframes)))
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)

        for t in batch:
            rgb, mask, person_mask = keyframe_data[t]
            loss = _canonical_frame_loss(
                scene, pose_params, t, rgb, mask, viewmat, K, H, W, cfg,
                person_mask=person_mask,
            )
            batch_loss = batch_loss + loss / len(batch)

        if cfg.weight_anchor > 0 and scene._anchor_positions is not None:
            batch_loss = batch_loss + cfg.weight_anchor * loss_anchor(
                scene.positions, scene._anchor_positions
            )

        batch_loss.backward()

        if scene._positions.grad is not None:
            pos_grad_accum += scene._positions.grad.norm(dim=-1).detach()

        optimizer.step()
        _apply_lr_decay(optimizer, i, cfg.n_batches, cfg)

        if cfg.reset_opacity_every > 0 and (i + 1) % cfg.reset_opacity_every == 0:
            with torch.no_grad():
                scene._opacities_raw.data.fill_(-3.0)

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
            optimizer = _canonical_optimizer(scene, pose_params, cfg)

        if (i + 1) % cfg.render_interval == 0 or i == 0:
            print(f"  batch {i+1:>5}/{cfg.n_batches}  "
                  f"loss={batch_loss.item():.4f}  N={scene.num_gaussians}")
            t_s = random.choice(keyframes)
            _save_canonical_debug(scene, pose_params, t_s, keyframe_data[t_s],
                                  viewmat, K, H, W, i + 1, output_dir, cfg)

    return scene


# --------------------------------------------------------------------------- #
# Render-and-compare tracking (Phase 2) — pose-only optimizer
# --------------------------------------------------------------------------- #

def _pose_optimizer_track(
    pose_params: ObjectPoseParams,
    cfg: RenderRefineConfig,
) -> torch.optim.Adam:
    return torch.optim.Adam([
        {'params': [pose_params.rotations_6d], 'lr': cfg.lr_pose_track},
        {'params': [pose_params.translations],  'lr': cfg.lr_pose_track},
    ], lr=0.0, eps=1e-15)


def _track_frame_loss(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    frame_t: int,
    rgb: torch.Tensor,
    mask: Optional[torch.Tensor],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    cfg: RenderRefineConfig,
    person_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Render-and-compare loss for single-frame pose optimisation."""
    R, t_vec = pose_params.get_transform(frame_t, 0)
    world_pos = scene.positions @ R.T + t_vec.unsqueeze(0)
    result = render(scene, world_pos, viewmat, K, H, W, sh_degree=cfg.sh_degree)

    m = mask.unsqueeze(-1) if mask is not None else torch.ones_like(rgb[:, :, :1])
    rgb_loss  = F.l1_loss(result.rgb * m, rgb * m)
    ssim_loss = loss_ssim(result.rgb * m, rgb * m)
    total = cfg.track_weight_rgb * rgb_loss + cfg.track_weight_ssim * ssim_loss

    if cfg.track_weight_mask > 0 and mask is not None:
        alpha = result.alpha.clamp(1e-6, 1 - 1e-6)
        per_pixel = F.binary_cross_entropy(alpha, mask, reduction='none')
        w = torch.where(mask > 0.5,
                        torch.ones_like(mask),
                        torch.full_like(mask, cfg.track_mask_outside_weight))
        if person_mask is not None:
            occluded = (mask < 0.5) & (person_mask > 0.5)
            w = torch.where(occluded, torch.full_like(w, cfg.track_mask_occluder_weight), w)
        if cfg.track_mask_focal_gamma > 0:
            with torch.no_grad():
                p_t = torch.where(mask > 0.5, alpha, 1.0 - alpha)
                focal_w = (1.0 - p_t) ** cfg.track_mask_focal_gamma
            mask_loss = (focal_w * per_pixel * w).mean()
        else:
            mask_loss = (per_pixel * w).mean()
        total = total + cfg.track_weight_mask * mask_loss

    return total


def _plateau(losses: list, patience: int, threshold: float) -> bool:
    if len(losses) < patience:
        return False
    improvement = (losses[-patience] - losses[-1]) / (abs(losses[-patience]) + 1e-8)
    return improvement < threshold


def track_all_frames(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    frame_data: Dict[int, Tuple],
    frame_indices: List[int],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    cfg: RenderRefineConfig,
    output_dir: str,
) -> None:
    """
    Phase 2: per-frame pose optimisation with frozen canonical.

    Processes frames in temporal order.  Each frame's pose is warm-started
    from the previous refined pose, then optimised until the render-vs-observe
    loss plateaus or the step budget is exhausted.
    """
    sorted_frames = sorted(frame_indices)
    print(f"[render_refine] Tracking {len(sorted_frames)} frames "
          f"(max {cfg.n_steps_track} steps each)…\n")

    for frame_idx, t in enumerate(sorted_frames):
        rgb, mask, person_mask = frame_data[t]

        # Warm-start pose from previous refined frame
        if frame_idx > 0:
            prev_t = sorted_frames[frame_idx - 1]
            with torch.no_grad():
                pose_params.rotations_6d[t, 0] = pose_params.rotations_6d[prev_t, 0].clone()
                pose_params.translations[t, 0]  = pose_params.translations[prev_t, 0].clone()

        optimizer = _pose_optimizer_track(pose_params, cfg)
        losses: list = []
        ema = None
        loss = torch.tensor(0.0)
        step = 0

        for step in range(cfg.n_steps_track):
            optimizer.zero_grad()
            loss = _track_frame_loss(
                scene, pose_params, t, rgb, mask, viewmat, K, H, W, cfg,
                person_mask=person_mask,
            )
            loss.backward()
            optimizer.step()

            v = loss.item()
            ema = v if ema is None else cfg.track_ema_alpha * v + (1 - cfg.track_ema_alpha) * ema
            losses.append(ema)

            if step >= cfg.track_min_steps and _plateau(
                losses, cfg.track_plateau_patience, cfg.track_plateau_threshold
            ):
                break

        print(f"  frame {frame_idx+1:>4}/{len(sorted_frames)}  t={t}  "
              f"loss={loss.item():.4f}  steps={step+1}")

        if (frame_idx + 1) % cfg.track_debug_every == 0 or frame_idx == 0:
            with torch.no_grad():
                _save_track_debug(scene, pose_params, t, frame_data[t],
                                  viewmat, K, H, W, output_dir, cfg)


# --------------------------------------------------------------------------- #
# Debug renders
# --------------------------------------------------------------------------- #

def _render_debug_pair(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    frame_t: int,
    frame_entry: Tuple,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    cfg: RenderRefineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render scene at frame_t pose; return (original_bgr, rendered_bgr) uint8."""
    rgb, mask, _ = frame_entry
    R, t_vec = pose_params.get_transform(frame_t, 0)
    world_pos = scene.positions @ R.T + t_vec.unsqueeze(0)
    result = render(scene, world_pos, viewmat, K, H, W, sh_degree=cfg.sh_degree)

    rendered_bgr = cv2.cvtColor(
        (result.rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    original_bgr = cv2.cvtColor(
        (rgb.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    if mask is not None:
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_bgr, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(rendered_bgr, contours, -1, (0, 255, 0), 2)

    _draw_pose_axes(rendered_bgr, R, world_pos.mean(0), K)
    return original_bgr, rendered_bgr


def _save_canonical_debug(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    frame_t: int,
    frame_entry: Tuple,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    batch_idx: int,
    output_dir: str,
    cfg: RenderRefineConfig,
) -> None:
    debug_dir = os.path.join(output_dir, 'debug_canonical')
    os.makedirs(debug_dir, exist_ok=True)
    with torch.no_grad():
        orig, rend = _render_debug_pair(scene, pose_params, frame_t, frame_entry,
                                        viewmat, K, H, W, cfg)
    combined = np.concatenate([orig, rend], axis=1)
    cv2.putText(combined, f'canonical  batch {batch_idx}  frame {frame_t}', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(debug_dir, f'b{batch_idx:06d}_f{frame_t:06d}.png'), combined)


def _save_track_debug(
    scene: GaussianScene,
    pose_params: ObjectPoseParams,
    frame_t: int,
    frame_entry: Tuple,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    output_dir: str,
    cfg: RenderRefineConfig,
) -> None:
    debug_dir = os.path.join(output_dir, 'debug_track')
    os.makedirs(debug_dir, exist_ok=True)
    orig, rend = _render_debug_pair(scene, pose_params, frame_t, frame_entry,
                                    viewmat, K, H, W, cfg)
    combined = np.concatenate([orig, rend], axis=1)
    cv2.putText(combined, f'track  frame {frame_t}', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(debug_dir, f'{frame_t:06d}.png'), combined)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def gsplat_render_refine(
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
    for d in ('debug_canonical', 'debug_track'):
        dp = os.path.join(output_dir, d)
        if os.path.exists(dp):
            shutil.rmtree(dp)

    # ------------------------------------------------------------------
    # Frames + intrinsics
    # ------------------------------------------------------------------
    frame_indices = _find_frame_indices(images_dir, frame_step)
    if not frame_indices:
        raise RuntimeError(f"No frames found in {images_dir}")
    n_frames = max(frame_indices) + 1
    print(f"[render_refine] {len(frame_indices)} frames  device={device}")

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

    # ------------------------------------------------------------------
    # Load frames
    # ------------------------------------------------------------------
    print("[render_refine] Loading frames…")
    frame_data = load_frame_data(images_dir, masks_dir, frame_indices, s, device,
                                 person_masks_dir=person_masks_dir)

    # ------------------------------------------------------------------
    # Initialise scene + FP poses
    # ------------------------------------------------------------------
    # Build a minimal config compatible with init_scene's type annotation
    init_cfg = _OC()
    init_cfg.n_gaussians    = cfg.n_gaussians
    init_cfg.initial_opacity = cfg.initial_opacity
    init_cfg.device         = cfg.device

    scene = init_scene(mesh_path, images_dir, depth_dir, masks_dir,
                       intrinsics_full, frame_indices, poses_dir, init_cfg).to(device)

    pose_params = ObjectPoseParams(n_frames, num_objects=1, device=device)
    init_poses(pose_params, poses_dir, frame_indices)

    # ------------------------------------------------------------------
    # Phase 1a — select keyframes
    # ------------------------------------------------------------------
    print("[render_refine] Selecting keyframes (temporal consistency + mesh IoU)…")
    keyframe_indices = select_keyframes(
        pose_params, frame_indices, cfg,
        frame_data=frame_data,
        mesh_path=mesh_path,
        poses_dir=poses_dir,
        intrinsics_train=intrinsics_train,
    )
    print(f"[render_refine] {len(keyframe_indices)} keyframes selected  "
          f"(frames {keyframe_indices[0]}…{keyframe_indices[-1]})")

    with open(os.path.join(output_dir, 'keyframes.txt'), 'w') as f:
        f.write('\n'.join(str(t) for t in keyframe_indices))

    keyframe_data = {t: frame_data[t] for t in keyframe_indices if t in frame_data}

    # ------------------------------------------------------------------
    # Phase 1b — train canonical with fixed poses
    # ------------------------------------------------------------------
    scene = train_canonical(
        scene, pose_params, keyframe_data, viewmat, K, H, W, cfg, output_dir
    )

    print("\n[render_refine] Saving canonical…")
    save_gaussians_ply(scene, os.path.join(output_dir, 'canonical_gaussians.ply'))

    # ------------------------------------------------------------------
    # Phase 2 — render-and-compare tracking
    # Re-initialise poses from FP so tracking starts from the same prior
    # as Phase 1 (not from any drift introduced during canonical training).
    # ------------------------------------------------------------------
    print("\n[render_refine] Phase 2: re-initialising poses from FP…")
    pose_params = ObjectPoseParams(n_frames, num_objects=1, device=device)
    init_poses(pose_params, poses_dir, frame_indices)

    # Freeze canonical — only pose is optimised in Phase 2
    for p in scene.parameters():
        p.requires_grad_(False)

    track_all_frames(
        scene, pose_params, frame_data, frame_indices,
        viewmat, K, H, W, cfg, output_dir,
    )

    # ------------------------------------------------------------------
    # Save refined poses
    # ------------------------------------------------------------------
    print("\n[render_refine] Saving refined poses…")
    _save_poses(pose_params, frame_indices, output_dir)
    print(f"[render_refine] Done.  Outputs at: {output_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Two-phase render-and-compare Gaussian object tracking'
    )
    parser.add_argument('--images_dir',       required=True)
    parser.add_argument('--masks_dir',        required=True)
    parser.add_argument('--intrinsics_path',  required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--config_path',      default=None)
    parser.add_argument('--mesh_path',        default=None)
    parser.add_argument('--depth_dir',        default=None)
    parser.add_argument('--poses_dir',        default=None)
    parser.add_argument('--person_masks_dir', default=None)
    parser.add_argument('--frame_step',       type=int, default=1)
    args = parser.parse_args()

    gsplat_render_refine(
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
