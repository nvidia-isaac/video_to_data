"""Joint hand+object Gaussian-splatting refinement.

Optimizes:
  Globals (shared across all frames):
    - Object Gaussian attributes  (Δp, q, log-scale, opacity, color)
    - Per-hand Gaussian attributes (isotropic log-scale, opacity, color)

  Per-frame:
    - Object pose (axis-angle, translation)
    - MANO global_orient, hand_pose, cam_t for each hand track

Losses (all jointly, no warmup):
    - Masked photometric L1 (combined render vs. image, in union mask)
    - Per-set silhouette L1 (object alpha vs. object mask; each hand alpha
      vs. that hand's mask)
    - Optional MoGe depth Huber (when ``depth_dir`` is provided)
    - Temporal smoothness on per-frame pose params

Inputs are loaded with ``v2d.gsplat_refinement.lib.io`` and outputs are
written back in the same on-disk formats so existing renderers (e.g.
``v2d.hamer.docker.run_render_hands_aligned_video``) work unchanged on the
refined data.

Usage:
    python -m v2d.gsplat_refinement.lib.refine \\
        --frames_dir              /data/frames \\
        --intrinsics_path         /data/intrinsics.json \\
        --object_mesh_path        /data/mesh.obj \\
        --object_poses_dir        /data/poses_smoothed \\
        --object_mask_dir         /data/object_masks \\
        --right_hand_pose_dir     /data/hamer_aligned/2 \\
        --right_hand_mask_dir     /data/masks/2 \\
        --mano_assets_root        /data/weights/hamer/_DATA/data \\
        --refined_object_poses_dir /data/refined_poses \\
        --refined_right_hand_pose_dir /data/refined_hamer/2 \\
        --overlay_path            /data/refined_overlay.mp4
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .background import (
    BackgroundGaussians,
    BackgroundPoseField,
    init_background_from_depth,
    init_background_multiframe,
)
from .gaussians import (
    concat_frames,
    init_hand_gaussians,
    init_object_gaussians_from_mesh,
    resample_mesh_surface,
)
from .io import (
    FrameCache,
    HandPoseTrack,
    ObjectPoseTrack,
    list_frame_indices,
    load_hand_poses,
    load_intrinsics,
    load_object_mesh,
    load_object_poses,
    load_rgb,
    save_hand_poses,
    save_object_poses,
)
from .losses import (
    balanced_photometric_loss,
    beta_prior_loss,
    delta_p_regularizer,
    depth_loss,
    hand_scale_prior_loss,
    photometric_loss,
    pose_init_prior_loss,
    rotation_smoothness,
    silhouette_loss,
    temporal_smoothness,
)
from .pose_fields import HandPoseField, ObjectPoseField
from .render import render_rgb_depth


# ---------------------------------------------------------------------------
# Background pose seeding (external VO/SfM init)
# ---------------------------------------------------------------------------


def _load_relative_w2c_poses(
    poses_dir: str,
    frame_indices: list[int],
    ref_t: int,
) -> dict[int, np.ndarray]:
    """Load per-frame Transform3d JSONs and return relative world→camera SE(3).

    gsplat's "world" is the reference frame's camera frame, so we rebase the
    DROID/COLMAP cam-to-world poses against the reference frame:

        T_w2c_gsplat(t) = T_cw(t)^-1 @ T_cw(ref)

    Returns a dict mapping ``positional index t`` → 4×4 ``np.float64`` matrix.
    The ref frame is always present and set to identity. Frames whose JSON
    is missing OR whose ``Transform3d.scale`` differs from 1 are omitted from
    the result (caller can treat them as "no SLAM info").
    """
    from v2d.common.datatypes import Transform3d

    ref_fidx = frame_indices[ref_t]
    ref_path = os.path.join(poses_dir, f"{ref_fidx:06d}.json")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"background_pose_init_dir is set but reference-frame pose "
            f"{ref_path} does not exist. The relative-pose composition needs "
            f"the reference frame to anchor the gsplat world frame."
        )
    M_ref = Transform3d.load(ref_path).to_matrix()         # cam→world (DROID)

    out: dict[int, np.ndarray] = {ref_t: np.eye(4)}
    for t, fidx in enumerate(frame_indices):
        if t == ref_t:
            continue
        path = os.path.join(poses_dir, f"{fidx:06d}.json")
        if not os.path.exists(path):
            continue
        tf = Transform3d.load(path)
        if any(abs(s - 1.0) > 1e-3 for s in tf.scale):
            continue
        M_cw_t = tf.to_matrix()
        out[t] = np.linalg.inv(M_cw_t) @ M_ref
    return out


def _seed_background_pose_field(
    bg_pose_field,                          # BackgroundPoseField
    poses: dict[int, np.ndarray],
) -> tuple[int, int]:
    """Overwrite bg_pose_field.{axis_angle, translation} from a pre-loaded
    dict of positional index → T_w2c_gsplat 4×4 matrix.

    Missing positions keep the identity (zeros) init that BackgroundPoseField
    starts with.
    """
    from .gaussians import rotmat_to_quat
    from .pose_fields import _quat_to_axis_angle

    device = bg_pose_field.axis_angle.device
    aa_init = bg_pose_field.axis_angle.detach().clone()
    tr_init = bg_pose_field.translation.detach().clone()

    T = aa_init.shape[0]
    n_loaded = 0
    n_missing = 0
    for t in range(T):
        if t not in poses:
            n_missing += 1
            continue
        M = poses[t]
        R = torch.from_numpy(M[:3, :3]).to(device, dtype=torch.float32)
        t_vec = torch.from_numpy(M[:3, 3]).to(device, dtype=torch.float32)
        q = rotmat_to_quat(R)
        aa = _quat_to_axis_angle(q)
        aa_init[t] = aa
        tr_init[t] = t_vec
        n_loaded += 1

    with torch.no_grad():
        bg_pose_field.axis_angle.data.copy_(aa_init)
        bg_pose_field.translation.data.copy_(tr_init)
    return n_loaded, n_missing


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class HandSlot:
    """All per-hand state bundled together: input data + learnable modules."""
    side: str                   # "left" or "right"
    track: HandPoseTrack
    pose_field: HandPoseField
    gaussians: object           # HandGaussians (avoid circular import in type)
    mask_dir: str
    output_pose_dir: str | None


@dataclass
class LossWeights:
    photometric: float = 1.0
    silhouette:  float = 0.5
    # Per-class relative weights inside the silhouette loss. When SAM2
    # hand masks are noisier than the object mask (typical), drop
    # silhouette_hand to e.g. 0.3 to trust object more without changing
    # the overall silhouette/photometric balance.
    silhouette_obj:  float = 1.0
    silhouette_hand: float = 1.0
    depth:       float = 0.05
    # Smoothness terms use .sum() reduction in losses.py — gradients are
    # length-invariant, so weights here are much smaller than under the
    # old .mean() formulation. Rotation and translation are split so
    # either can be disabled or weighted independently (e.g. set
    # smooth_obj_rot=0 to allow free per-frame rotation while still
    # smoothing translation).
    smooth_obj_rot:    float = 0.01
    smooth_obj_trans:  float = 0.01
    # Hand rotation is split into global vs finger because finger joints
    # carry much weaker photometric signal per joint (a fingertip covers
    # ~10 pixels vs. the whole hand orientation affecting hundreds), so a
    # smoothness that's right for global_orient over-smooths fingers and
    # kills articulation. Default finger weight is 10× smaller.
    smooth_hand_rot:    float = 0.01    # global_orient (T, 3)
    smooth_hand_finger: float = 0.001   # hand_pose (T, 15, 3) — 15 joints
    smooth_hand_trans:  float = 0.01    # cam_t (T, 3)
    # Background camera motion is usually slow and continuous; tighter
    # smoothness than foreground keeps the optimizer from absorbing
    # photometric noise as fake camera jitter.
    smooth_bg_rot:      float = 0.1
    smooth_bg_trans:    float = 0.1
    beta_prior:  float = 10.0      # tight; keeps β from absorbing per-frame photometric noise
    hand_scale_prior: float = 10.0  # tight by default — align_hands' median is a good init
    # Δp regularizer also uses .sum() now: per-Gaussian gradient is no
    # longer divided by N. Default 1.0 with N≈1000 gives a comparable
    # pull to a photometric loss of ~0.1 magnitude.
    delta_p_reg: float = 1.0
    # Tight Gaussian prior on log_scale_global (pulls global object scale
    # toward 1.0). Drop to ~0 if you want unrestricted scale; raise to
    # freeze it near init. Default is moderate — prior is observable enough
    # via background depth / hand size to converge without tight pinning.
    obj_scale_prior: float = 1.0


# ---------------------------------------------------------------------------
# Frame index alignment
# ---------------------------------------------------------------------------

def _common_frame_indices(*lists: list[int]) -> list[int]:
    """Intersection of multiple frame-index lists, preserving order."""
    common = set(lists[0])
    for lst in lists[1:]:
        common &= set(lst)
    return sorted(common)


def _restrict_object_track(track: ObjectPoseTrack, keep: set[int]) -> ObjectPoseTrack:
    idxs = [i for i, fi in enumerate(track.frame_indices) if fi in keep]
    return ObjectPoseTrack(
        rotations    = track.rotations[idxs],
        translations = track.translations[idxs],
        scales       = track.scales[idxs],
        frame_indices= [track.frame_indices[i] for i in idxs],
    )


def _restrict_hand_track(track: HandPoseTrack, keep: set[int]) -> HandPoseTrack:
    idxs = [i for i, fi in enumerate(track.frame_indices) if fi in keep]
    return HandPoseTrack(
        global_orient = track.global_orient[idxs],
        hand_pose     = track.hand_pose[idxs],
        betas         = track.betas[idxs],
        cam_t         = track.cam_t[idxs],
        is_right      = track.is_right,
        frame_indices = [track.frame_indices[i] for i in idxs],
        raw_records   = [track.raw_records[i] for i in idxs],
    )


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def _save_overlay_video(
    overlay_frames: list[np.ndarray],   # list of (H, W, 3) uint8
    output_path: str,
    fps: float = 30.0,
) -> None:
    if not overlay_frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(overlay_frames):
            Image.fromarray(fr).save(os.path.join(tmp, f"{i:06d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmp, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ], check=True)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def refine(
    frames_dir: str,
    intrinsics_path: str,
    object_mesh_path: str,
    object_poses_dir: str,
    object_mask_dir: str,
    refined_object_poses_dir: str,
    overlay_path: str,
    # Learned global object scale: when provided, written as a JSON file
    # ``{"scale": float}``. Per-frame Transform3d scales are then left
    # untouched (passthrough of input). When None, falls back to baking
    # the learned scale into each frame's Transform3d.scale (legacy).
    refined_object_scale_path: str | None = None,
    left_hand_pose_dir: str | None = None,
    left_hand_mask_dir: str | None = None,
    right_hand_pose_dir: str | None = None,
    right_hand_mask_dir: str | None = None,
    refined_left_hand_pose_dir: str | None = None,
    refined_right_hand_pose_dir: str | None = None,
    depth_dir: str | None = None,
    mano_assets_root: str | None = None,
    n_epochs: int = 30,
    n_gaussian_only_epochs: int = 5,           # epochs at the start with poses frozen
    batch_size: int = 4,
    lr_gaussians: float = 1e-2,
    lr_hand_gaussians: float | None = None,    # defaults to lr_gaussians if None
    # Per-attribute LR multipliers applied on top of the per-set base LR
    # (lr_gaussians / lr_hand_gaussians / lr_bg_gaussians). Defaults to 1.0
    # everywhere preserves the old single-LR behavior; standard 3DGS uses
    # very different ratios (e.g. opacity ~300x position) — see docstrings
    # on the CLI flags for typical settings.
    lr_mul_delta_p: float = 1.0,
    lr_mul_quat: float = 1.0,
    lr_mul_scale: float = 1.0,
    lr_mul_opacity: float = 1.0,
    lr_mul_color: float = 1.0,
    lr_mul_obj_global_scale: float = 1.0,
    lr_object_pose: float = 1e-3,                # legacy: applies to both rot+trans if specifics are None
    lr_object_rot: float | None = None,          # axis_angle; falls back to lr_object_pose
    lr_object_trans: float | None = None,        # translation; falls back to lr_object_pose
    lr_hand_pose: float = 1e-3,                  # legacy: applies to global_orient + hand_pose + cam_t if specifics are None
    lr_hand_global_orient: float | None = None,  # MANO root rotation; falls back to lr_hand_pose
    lr_hand_finger: float | None = None,         # MANO hand_pose (15 joint axis-angles); falls back to lr_hand_pose
    lr_hand_trans: float | None = None,          # MANO cam_t; falls back to lr_hand_pose
    lr_betas: float = 1e-4,
    # Optional optimization of per-track hand_scale (init from align_hands).
    # When False (default), hand_scale stays fixed at the align-hands estimate
    # and is excluded from the optimizer. Prior weight lives on LossWeights
    # (``hand_scale_prior``), wired through the lib CLI.
    learn_hand_scale: bool = False,
    lr_hand_scale:    float = 1e-3,
    render_every: int = 0,
    progress_dir: str | None = None,
    debug_frame_idx: int | None = None,
    mask_background_to_black: bool = False,
    balance_photometric_by_mask: bool = False,
    freeze_object_rot: bool = False,
    freeze_object_trans: bool = False,
    freeze_object_scale: bool = False,
    freeze_hand_rot: bool = False,
    freeze_hand_trans: bool = False,
    with_background: bool = False,
    bg_ref_frame: int | None = None,            # ref frame index; defaults to debug_frame_idx or first frame
    lr_bg_gaussians: float | None = None,       # defaults to lr_gaussians
    lr_bg_pose: float = 1e-3,                    # legacy: applies to both rot+trans if specifics are None
    lr_bg_rot: float | None = None,              # axis_angle; falls back to lr_bg_pose
    lr_bg_trans: float | None = None,            # translation; falls back to lr_bg_pose
    bg_max_points: int = 50000,
    # Optional: seed the background pose field with an external VO/SfM solution
    # (e.g. DROID-SLAM). Folder of per-frame `Transform3d` JSONs encoding
    # camera-to-world SE(3) (DROID/COLMAP convention). When set, after
    # BackgroundPoseField construction we replace the identity init with
    # T_w2c_gsplat(t) = T_cw(t)^-1 @ T_cw(ref) so the ref frame stays identity
    # and other frames carry the relative motion. Missing/unreadable frames
    # fall back to identity. Frames whose `Transform3d.scale` differs from 1
    # are ignored — pass a scale-aligned trajectory (e.g. via DROID's
    # --align_to_depth_folder).
    background_pose_init_dir: str | None = None,
    # Multi-frame BG point-cloud init (only effective when
    # background_pose_init_dir is also set, since we need per-frame poses to
    # compose unprojections into a common world frame). When stride > 1, the
    # BG anchor cloud is fused from MoGe depth at frame indices [ref, ref+s,
    # ref+2s, ...] ∪ [ref-s, ref-2s, ...] using each frame's own SAM2
    # foreground mask. Voxel-downsampled to bg_voxel_size before final
    # random subsample to bg_max_points. Stride=1 or =0 → single-frame init.
    bg_init_stride: int   = 10,
    bg_voxel_size:  float = 0.005,
    n_obj_gaussians: int | None = None,    # None = use mesh vertex count as-is
    n_hand_gaussians: int | None = None,   # None = use 778 (full MANO verts);
                                             # smaller subsamples; >778 ignored
                                             # (would require barycentric LBS).
    use_cosine_lr_schedule: bool = False,
    cosine_lr_min_ratio: float = 0.0,      # final LR / initial LR (per group)
    # Coarse-to-fine scale annealing: at training start, multiply every
    # Gaussian's render-time scale by this factor; decay log-linearly to
    # 1.0 over ``coarse_decay_epochs`` (defaults to n_epochs). 1.0 disables.
    # Wider Gaussians mean even a far-off pose has photometric overlap
    # with the target — helps recover outlier frames that would otherwise
    # be stuck in flat regions of the loss.
    coarse_init_scale_factor: float = 1.0,
    coarse_decay_epochs: int | None = None,
    # Distance-from-reference per-frame pose confidence. c_t scales
    # per-frame photometric/silhouette/depth (high-c frames contribute
    # more to Gaussian appearance) AND modulates a pose-init prior
    # (high-c frames are pinned near input pose). 0 disables.
    pose_confidence_decay: float = 0.0,                     # τ in frames; 0 disables (static c)
    pose_confidence_ref_frame: int | None = None,           # fidx; falls back to bg_ref_frame, then first frame
    # Dynamic per-frame confidence based on quaternion-aligned distance
    # to neighbor frames' rotations, recomputed every batch from the
    # current pose state. Multiplies into the per-frame loss together
    # with the static confidence: c_total[t] = c_static[t] * c_dynamic[t].
    # Set the static decay to 0 to use ONLY dynamic (uniform static = 1).
    pose_confidence_dynamic_tau: float = 0.0,               # quat-distance² scale; 0 disables
    w_pose_init_prior: float = 0.0,                         # weight for the c_t-scaled pose-init prior
    # Per-frame discrete rotation search for the object pose. For each
    # frame, render the current Gaussian set at N candidate rotations
    # (batched as multiple viewmats in a single gsplat call), score by
    # photometric + silhouette IoU vs the SAM2 target, snap the frame's
    # axis_angle to the best candidate. Handles rotation errors that are
    # too large for the photometric basin to recover via gradient.
    rotation_search_n_candidates: int = 0,                  # 0 disables
    rotation_search_period: int = 0,                        # 0 = run once at start; >0 = also every K epochs
    rotation_search_local_frac: float = 0.5,                # fraction of candidates as local perturbations (rest are global SO(3))
    rotation_search_local_max_deg: float = 30.0,            # max angle for local perturbations
    rotation_search_silhouette_weight: float = 1.0,         # IoU weight in scoring (photometric weight is 1.0)
    rotation_search_smoothness_weight: float = 1.0,         # causal smoothness weight: penalize candidates far from previous frame's rotation (in quat-aligned space). Forward sweep is implicit; t=0 has no penalty.
    use_l2_photometric: bool = False,                       # squared error instead of L1 for the photometric loss
    use_l2_silhouette: bool = False,                        # squared error instead of L1 for the class-label silhouette loss
    # Train at lower resolution for speed. 1.0 = native; 0.5 ≈ 4x faster
    # rendering (gsplat is ~linear in pixel count). Cache, intrinsics,
    # and all downstream renders use the scaled dimensions; the final
    # overlay video is also at the scaled resolution.
    train_resolution_scale: float = 1.0,
    multiview_include_background: bool = False,    # include bg Gaussians in the final multi-view grid
    # Checkpointing: save a single .pt with all module state_dicts +
    # optimizer + LR scheduler + init pose snapshot + step counter, so
    # a follow-up run can resume from exactly this state and run more
    # epochs. ``checkpoint_path`` controls where to save (None disables);
    # ``resume_from_checkpoint`` loads such a file at startup before the
    # epoch loop.
    checkpoint_path: str | None = None,
    resume_from_checkpoint: str | None = None,
    ignore_optimizer_state: bool = False,    # when resuming, skip optimizer/scheduler state load
    freeze_gaussians: bool = False,          # freeze ALL Gaussian attrs (obj + hand + bg)
    random_init_obj_pose: bool = False,      # randomize obj pose after build (uniform rot, σ=0.1 trans)
    random_init_obj_pose_trans_std: float = 0.1,
    freeze_bg_rot: bool = False,
    freeze_bg_trans: bool = False,
    weights: LossWeights | None = None,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    if weights is None:
        weights = LossWeights()
    if lr_hand_gaussians is None:
        lr_hand_gaussians = lr_gaussians
    if lr_bg_gaussians is None:
        lr_bg_gaussians = lr_gaussians
    # Resolve per-DOF pose LR overrides; default to the lumped legacy LR.
    if lr_object_rot is None:           lr_object_rot          = lr_object_pose
    if lr_object_trans is None:         lr_object_trans        = lr_object_pose
    if lr_hand_global_orient is None:   lr_hand_global_orient  = lr_hand_pose
    if lr_hand_finger is None:          lr_hand_finger         = lr_hand_pose
    if lr_hand_trans is None:           lr_hand_trans          = lr_hand_pose
    if lr_bg_rot is None:               lr_bg_rot              = lr_bg_pose
    if lr_bg_trans is None:             lr_bg_trans            = lr_bg_pose
    if (left_hand_pose_dir is not None or right_hand_pose_dir is not None) and \
       mano_assets_root is None:
        raise ValueError("mano_assets_root is required when any hand pose dir is set")
    if with_background and depth_dir is None:
        raise ValueError(
            "with_background=True requires depth_dir for reference-frame "
            "depth unprojection. Pass --depth_dir."
        )

    torch.manual_seed(seed)
    device = torch.device(device)

    # ----- intrinsics, frames, masks ----------------------------------------
    K, W, H = load_intrinsics(intrinsics_path, str(device))
    if train_resolution_scale != 1.0:
        s = float(train_resolution_scale)
        K = K.clone()
        K[0, 0] *= s   # fx
        K[1, 1] *= s   # fy
        K[0, 2] *= s   # cx
        K[1, 2] *= s   # cy
        W_scaled = max(1, int(round(W * s)))
        H_scaled = max(1, int(round(H * s)))
        print(f"Training resolution: {W}x{H} → {W_scaled}x{H_scaled} "
              f"(scale={s:.3f}); render time ≈ {s*s:.3f}× of native.")
        W, H = W_scaled, H_scaled

    object_track = load_object_poses(object_poses_dir, str(device))
    hand_tracks: list[tuple[str, HandPoseTrack, str, str | None]] = []
    if right_hand_pose_dir is not None:
        if right_hand_mask_dir is None:
            raise ValueError("right_hand_mask_dir is required with right_hand_pose_dir")
        hand_tracks.append(("right", load_hand_poses(right_hand_pose_dir, str(device)),
                            right_hand_mask_dir, refined_right_hand_pose_dir))
    if left_hand_pose_dir is not None:
        if left_hand_mask_dir is None:
            raise ValueError("left_hand_mask_dir is required with left_hand_pose_dir")
        hand_tracks.append(("left", load_hand_poses(left_hand_pose_dir, str(device)),
                            left_hand_mask_dir, refined_left_hand_pose_dir))

    # Restrict everything to frames present in *all* sources (frames + object
    # poses + every hand pose). Mask dirs aren't required to be complete; a
    # missing per-frame mask is treated as zero supervision for that frame.
    all_idx_sets: list[list[int]] = [
        list_frame_indices(frames_dir, ".png") or list_frame_indices(frames_dir, ".jpg"),
        object_track.frame_indices,
    ]
    for _, ht, _, _ in hand_tracks:
        all_idx_sets.append(ht.frame_indices)
    frame_indices = _common_frame_indices(*all_idx_sets)
    if not frame_indices:
        raise RuntimeError("No frames common to all input sources")
    print(f"Refining over {len(frame_indices)} frames "
          f"(intersection of {len(all_idx_sets)} sources)")

    object_track = _restrict_object_track(object_track, set(frame_indices))
    hand_tracks  = [
        (side, _restrict_hand_track(ht, set(frame_indices)), md, out)
        for side, ht, md, out in hand_tracks
    ]

    # ----- learnable state --------------------------------------------------
    obj_verts, obj_colors, _obj_faces = load_object_mesh(object_mesh_path, str(device))
    if n_obj_gaussians is not None and n_obj_gaussians != obj_verts.shape[0]:
        print(f"Object: resampling mesh surface to {n_obj_gaussians} "
              f"Gaussians (mesh has {obj_verts.shape[0]} verts).")
        obj_verts, obj_colors = resample_mesh_surface(
            obj_verts, obj_colors, _obj_faces, n_obj_gaussians,
        )
    obj_gaussians = init_object_gaussians_from_mesh(obj_verts, obj_colors).to(device)
    obj_pose_field = ObjectPoseField(object_track).to(device)

    if random_init_obj_pose:
        with torch.no_grad():
            n = obj_pose_field.axis_angle.shape[0]
            # Uniform random unit quaternion → axis_angle for each frame.
            q = torch.randn(n, 4, device=device)
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            from .pose_fields import _quat_to_axis_angle
            obj_pose_field.axis_angle.data = _quat_to_axis_angle(q)
            # Translation: add Gaussian noise to the input. Default std
            # is 0.1 m, typically the order of object size for HOI.
            noise = torch.randn_like(obj_pose_field.translation) * float(random_init_obj_pose_trans_std)
            obj_pose_field.translation.data = obj_pose_field.translation.data + noise
        print(f"Randomized object pose: uniform SO(3) rotation, "
              f"translation += N(0, {random_init_obj_pose_trans_std:.3f} m).")

    hand_slots: list[HandSlot] = []
    for side, ht, md, out_pose in hand_tracks:
        pf = HandPoseField(
            ht, mano_assets_root, device=device,
            learn_hand_scale=learn_hand_scale,
        ).to(device)
        # Optionally subsample the MANO vertex set to reduce hand Gaussian
        # count. Larger-than-778 not supported here (would require
        # barycentric LBS skinning).
        sub_idx: torch.Tensor | None = None
        if n_hand_gaussians is not None and n_hand_gaussians < pf.num_verts():
            g = torch.Generator(device="cpu").manual_seed(seed)
            sub_idx = torch.randperm(pf.num_verts(), generator=g)[:n_hand_gaussians]
            print(f"Hand ({side}): subsampling MANO verts "
                  f"{pf.num_verts()} → {n_hand_gaussians}.")
        elif n_hand_gaussians is not None and n_hand_gaussians > pf.num_verts():
            print(f"Hand ({side}): n_hand_gaussians={n_hand_gaussians} > "
                  f"{pf.num_verts()} MANO verts; using all 778. "
                  f"(Supersampling would require barycentric LBS.)")
        hg = init_hand_gaussians(
            n_verts           = pf.num_verts(),
            is_right          = ht.is_right,
            device            = device,
            subsample_indices = sub_idx,
        )
        hand_slots.append(HandSlot(
            side          = side,
            track         = ht,
            pose_field    = pf,
            gaussians     = hg,
            mask_dir      = md,
            output_pose_dir = out_pose,
        ))

    # ----- preload supervision into a CPU cache -----------------------------
    # Built before the optimizer because the optional background init needs
    # ref-frame RGB / depth / masks from this cache.
    cache = FrameCache(
        frame_indices    = frame_indices,
        frames_dir       = frames_dir,
        object_mask_dir  = object_mask_dir,
        hand_mask_dirs   = [s.mask_dir for s in hand_slots],
        depth_dir        = depth_dir,
        height           = H,
        width            = W,
    )

    # ----- optional background Gaussians + pose field -----------------------
    bg_gaussians: BackgroundGaussians | None = None
    bg_pose_field: BackgroundPoseField | None = None
    if with_background:
        # Resolve reference frame: prefer explicit bg_ref_frame, fall back
        # to debug_frame_idx (already validated to be in the sequence), then
        # to the first frame.
        if bg_ref_frame is not None:
            try:
                ref_t = frame_indices.index(int(bg_ref_frame))
            except ValueError:
                print(f"Warning: bg_ref_frame={bg_ref_frame} not in sequence; "
                      f"falling back to first frame.")
                ref_t = 0
        elif debug_frame_idx is not None:
            try:
                ref_t = frame_indices.index(int(debug_frame_idx))
            except ValueError:
                ref_t = 0
        else:
            ref_t = 0
        ref_fidx = frame_indices[ref_t]
        print(f"Initializing background from reference frame {ref_fidx:06d} "
              f"(positional t={ref_t})...")

        # Decide single-frame vs multi-frame init. Multi-frame requires
        # per-frame world→camera poses (= SLAM seeding) to compose
        # unprojections into a shared gsplat world. Without SLAM we can only
        # trust the reference frame.
        slam_poses: dict[int, np.ndarray] | None = None
        if background_pose_init_dir is not None:
            slam_poses = _load_relative_w2c_poses(
                background_pose_init_dir, frame_indices, ref_t,
            )

        def _union_mask_at(t: int) -> torch.Tensor:
            m = cache.obj_mask[t].to(device).clone()
            for hm in cache.hand_masks:
                m = torch.maximum(m, hm[t].to(device))
            return m

        K_intr = K
        use_multiframe = (
            slam_poses is not None
            and bg_init_stride is not None
            and bg_init_stride > 1
        )
        if use_multiframe:
            # Frames included: ref_t + every stride'th position before/after ref,
            # intersected with frames that have a valid SLAM pose.
            positions = sorted({ref_t} | set(range(0, len(frame_indices), bg_init_stride)))
            positions = [t for t in positions if t in slam_poses]
            print(f"Initializing background from {len(positions)} frames "
                  f"(ref t={ref_t}, stride={bg_init_stride}, voxel={bg_voxel_size} m)...")
            rgbs = [cache.rgb[t].to(device) for t in positions]
            depths = [cache.depth[t].to(device) for t in positions]
            masks  = [_union_mask_at(t)        for t in positions]
            T_w2c_list = [
                torch.from_numpy(slam_poses[t]).to(device, dtype=torch.float32)
                for t in positions
            ]
            anchors, colors, init_scales = init_background_multiframe(
                rgbs        = rgbs,
                depths      = depths,
                union_masks = masks,
                T_w2c_list  = T_w2c_list,
                K           = K_intr,
                voxel_size  = bg_voxel_size,
                max_points  = bg_max_points,
            )
        else:
            ref_fidx = frame_indices[ref_t]
            print(f"Initializing background from reference frame {ref_fidx:06d} "
                  f"(positional t={ref_t})...")
            anchors, colors, init_scales = init_background_from_depth(
                rgb         = cache.rgb[ref_t].to(device),
                depth       = cache.depth[ref_t].to(device),
                union_mask  = _union_mask_at(ref_t),
                K           = K_intr,
                max_points  = bg_max_points,
            )
        print(f"Background: {anchors.shape[0]} Gaussians initialized.")

        bg_gaussians  = BackgroundGaussians(anchors, colors, init_scales).to(device)
        bg_pose_field = BackgroundPoseField(
            n_frames=len(frame_indices), device=device, ref_frame_t=ref_t
        ).to(device)

        if slam_poses is not None:
            n_loaded, n_missing = _seed_background_pose_field(bg_pose_field, slam_poses)
            print(f"Background pose seeded from {background_pose_init_dir}: "
                  f"{n_loaded} frames loaded, {n_missing} fell back to identity.")

    # ----- optimizer --------------------------------------------------------
    # Per-attribute LR groups (multipliers shared across object / hand / bg
    # Gaussian sets — only the per-set base LR differs). Standard 3DGS uses
    # very different per-attribute ratios; making them user-tunable lets the
    # caller mimic those without touching the base LRs.
    def _gaussian_groups(gaussians, base_lr: float) -> list[dict]:
        groups = [
            {"params": [gaussians._delta_p],       "lr": base_lr * lr_mul_delta_p},
            {"params": [gaussians._quat_canon],    "lr": base_lr * lr_mul_quat},
            {"params": [gaussians._log_scale],     "lr": base_lr * lr_mul_scale},
            {"params": [gaussians._opacity_logit], "lr": base_lr * lr_mul_opacity},
            {"params": [gaussians._color],         "lr": base_lr * lr_mul_color},
        ]
        if hasattr(gaussians, "_log_scale_global"):
            groups.append({
                "params": [gaussians._log_scale_global],
                "lr": base_lr * lr_mul_obj_global_scale,
            })
        return groups

    # Per-DOF param groups so rotation, translation, and (for hands) finger
    # articulation can have independent learning rates. Adam's normalization
    # mostly compensates for gradient-magnitude differences, but rotation
    # needs many more steps per "physical unit" of motion than translation
    # — separate LRs let you boost rotation explicitly when poses are
    # converging slowly there.
    param_groups: list[dict] = []
    param_groups += _gaussian_groups(obj_gaussians, lr_gaussians)
    param_groups.append({"params": [obj_pose_field.axis_angle],  "lr": lr_object_rot})
    param_groups.append({"params": [obj_pose_field.translation], "lr": lr_object_trans})
    for slot in hand_slots:
        # Hand Gaussians get their own base LR knob — they typically need a bump
        # over the object's because the hand covers fewer pixels in image and
        # gradient signal per-Gaussian is correspondingly diluted.
        param_groups += _gaussian_groups(slot.gaussians, lr_hand_gaussians)
        param_groups.append({"params": [slot.pose_field.global_orient], "lr": lr_hand_global_orient})
        param_groups.append({"params": [slot.pose_field.hand_pose],     "lr": lr_hand_finger})
        param_groups.append({"params": [slot.pose_field.cam_t],         "lr": lr_hand_trans})
        param_groups.append({"params": [slot.pose_field.betas],         "lr": lr_betas})
        if learn_hand_scale:
            param_groups.append({"params": [slot.pose_field.hand_scale], "lr": lr_hand_scale})
    if bg_gaussians is not None:
        param_groups += _gaussian_groups(bg_gaussians, lr_bg_gaussians)
        param_groups.append({"params": [bg_pose_field.axis_angle],  "lr": lr_bg_rot})
        param_groups.append({"params": [bg_pose_field.translation], "lr": lr_bg_trans})
    optimizer = torch.optim.Adam(param_groups)

    # Optional cosine LR schedule. LambdaLR multiplies each param group's
    # *initial* LR by the same scalar each step, so per-attribute / per-set
    # LR ratios are preserved as the global scale decays.
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if use_cosine_lr_schedule:
        import math
        steps_per_epoch_local = (len(frame_indices) + batch_size - 1) // batch_size
        total_steps_local = max(1, n_epochs * steps_per_epoch_local)

        def _cosine_lambda(step: int) -> float:
            progress = min(step / max(1, total_steps_local - 1), 1.0)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_lr_min_ratio + (1.0 - cosine_lr_min_ratio) * cos_factor

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_lambda)
        print(f"Cosine LR schedule: decay over {total_steps_local} steps "
              f"to {cosine_lr_min_ratio:.2%} of initial LR.")

    # Per-class silhouette weights (allocated once, reused per step).
    silhouette_class_weights = torch.tensor(
        [weights.silhouette_obj] + [weights.silhouette_hand] * len(hand_slots),
        dtype=torch.float32, device=device,
    )

    # ----- pre-allocate static class-label tensor --------------------------
    # The per-Gaussian one-hot labels depend only on Gaussian counts, not
    # on per-frame state, so we build them once and reuse every step.
    # Layout matches concat_frames([obj] + hands + [bg]): block-diagonal
    # one-hots, with bg's block all zeros.
    K_classes_static = 1 + len(hand_slots)
    obj_n_static = obj_gaussians.num_gaussians()
    hand_n_static = [s.gaussians.num_gaussians() for s in hand_slots]
    bg_n_static  = bg_gaussians.num_gaussians() if bg_gaussians is not None else 0
    _label_chunks: list[torch.Tensor] = []
    obj_label = torch.zeros(obj_n_static, K_classes_static, device=device)
    obj_label[:, 0] = 1.0
    _label_chunks.append(obj_label)
    for i, n in enumerate(hand_n_static):
        h_label = torch.zeros(n, K_classes_static, device=device)
        h_label[:, i + 1] = 1.0
        _label_chunks.append(h_label)
    if bg_n_static > 0:
        _label_chunks.append(torch.zeros(bg_n_static, K_classes_static, device=device))
    labels_static = torch.cat(_label_chunks, dim=0).contiguous()             # (sum_N, K)

    # ----- per-frame pose confidence ----------------------------------------
    n_frames_local = len(frame_indices)
    if pose_confidence_decay > 0.0:
        # Resolve reference frame: explicit → bg_ref_frame → first frame.
        ref_t_pose = 0
        for cand in (pose_confidence_ref_frame, bg_ref_frame):
            if cand is None:
                continue
            try:
                ref_t_pose = frame_indices.index(int(cand))
                break
            except ValueError:
                continue
        t_arr = torch.arange(n_frames_local, device=device, dtype=torch.float32)
        pose_confidence = torch.exp(
            -(t_arr - float(ref_t_pose)).abs() / float(pose_confidence_decay)
        )
        print(f"Pose confidence: ref t={ref_t_pose} "
              f"(frame {frame_indices[ref_t_pose]:06d}), "
              f"τ={pose_confidence_decay} frames; "
              f"c range [{pose_confidence.min().item():.3f}, "
              f"{pose_confidence.max().item():.3f}].")
    else:
        pose_confidence = torch.ones(n_frames_local, device=device)

    # Snapshot of the initial object pose (frozen reference for the prior).
    init_obj_axis_angle  = obj_pose_field.axis_angle.detach().clone()
    init_obj_translation = obj_pose_field.translation.detach().clone()

    # Helper for dynamic pose confidence: per-frame quat-aligned distance
    # to neighbors → exp(-dist² / τ). Recomputed each batch.
    def _compute_dynamic_pose_confidence() -> torch.Tensor:
        with torch.no_grad():
            aa = obj_pose_field.axis_angle                              # (T, 3)
            T = aa.shape[0]
            ang = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            q = torch.cat([torch.cos(ang * 0.5),
                           aa / ang * torch.sin(ang * 0.5)], dim=-1)    # (T, 4)
            # Sign-aligned squared distance to prev (T-1 entries).
            dot = (q[1:] * q[:-1]).sum(dim=-1, keepdim=True)
            q_prev_aligned = torch.where(dot < 0, -q[:-1], q[:-1])
            d_prev = ((q[1:] - q_prev_aligned) ** 2).sum(dim=-1)        # (T-1,)
            # Build per-frame mean of available neighbor distances.
            zero = aa.new_zeros(1)
            d_to_prev = torch.cat([zero, d_prev])                       # (T,) — d to t-1
            d_to_next = torch.cat([d_prev, zero])                       # (T,) — d to t+1
            counts = torch.full((T,), 2.0, device=aa.device)
            counts[0]  = 1.0
            counts[-1] = 1.0
            mean_dist = (d_to_prev + d_to_next) / counts
            return torch.exp(-mean_dist / float(pose_confidence_dynamic_tau))

    # ----- training loop (epochs × random permutation × batched accumulation)
    n_frames = len(frame_indices)
    rng = np.random.default_rng(seed)
    steps_per_epoch = (n_frames + batch_size - 1) // batch_size
    total_steps = n_epochs * steps_per_epoch
    # Coarse-to-fine scale anneal setup: log-linear decay from
    # coarse_init_scale_factor → 1.0 over coarse_decay_steps.
    coarse_decay_steps = (
        (coarse_decay_epochs if coarse_decay_epochs is not None else n_epochs)
        * steps_per_epoch
    )
    if coarse_init_scale_factor != 1.0:
        print(f"Coarse-to-fine scale anneal: factor "
              f"{coarse_init_scale_factor:.2f} → 1.0 over "
              f"{coarse_decay_steps} steps.")
    print(f"Refining: {n_epochs} epochs x {steps_per_epoch} steps "
          f"(batch_size={batch_size}, n_frames={n_frames}, "
          f"~{n_epochs} updates/frame)")

    # ----- progress-render setup -------------------------------------------
    # When render_every > 0, dump an overlay PNG of a single fixed reference
    # frame to progress_dir at every Nth optimizer step. Cheap (one frame's
    # render) and lets you watch poses converge step-by-step.
    if render_every > 0:
        if progress_dir is None:
            progress_dir = os.path.join(
                os.path.dirname(os.path.abspath(overlay_path)) or ".", "progress"
            )
        os.makedirs(progress_dir, exist_ok=True)
        if debug_frame_idx is None:
            debug_frame_t = 0
        else:
            try:
                debug_frame_t = frame_indices.index(int(debug_frame_idx))
            except ValueError:
                debug_frame_t = 0
                print(f"Warning: debug_frame_idx={debug_frame_idx} not in "
                      f"the refined sequence; falling back to t=0 "
                      f"(frame {frame_indices[0]:06d}).")
        debug_fidx = frame_indices[debug_frame_t]
        print(f"Progress dumps every {render_every} steps → "
              f"{progress_dir}  (reference frame {debug_fidx:06d})")
    else:
        progress_dir = None
        debug_frame_t = None

    pbar = tqdm(total=total_steps, ncols=100, desc="refine")
    step_count = 0

    # ----- optional resume from checkpoint ---------------------------------
    # Loads after all modules + optimizer + scheduler are built, so
    # state_dict shapes are guaranteed to exist on the target side.
    if resume_from_checkpoint is not None:
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        if len(ckpt.get("hand_sides", [])) != len(hand_slots):
            raise ValueError(
                f"Checkpoint has {len(ckpt.get('hand_sides', []))} hand "
                f"slots; current run has {len(hand_slots)}."
            )
        ck_frames = ckpt.get("frame_indices")
        if ck_frames is not None and list(ck_frames) != list(frame_indices):
            print("Warning: checkpoint frame_indices differ from current "
                  "run — proceeding but state may not align.")

        obj_gaussians.load_state_dict(ckpt["obj_gaussians"])
        obj_pose_field.load_state_dict(ckpt["obj_pose_field"])
        for slot, hg, hp in zip(hand_slots,
                                ckpt["hand_gaussians"],
                                ckpt["hand_pose_fields"]):
            slot.gaussians.load_state_dict(hg)
            slot.pose_field.load_state_dict(hp)
        if bg_gaussians is not None and ckpt.get("bg_gaussians") is not None:
            bg_gaussians.load_state_dict(ckpt["bg_gaussians"])
            bg_pose_field.load_state_dict(ckpt["bg_pose_field"])
        if ignore_optimizer_state:
            print("Skipping optimizer/scheduler state on resume "
                  "(--ignore_optimizer_state); Adam will reinit lazily.")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            if lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "init_obj_axis_angle" in ckpt:
            init_obj_axis_angle  = ckpt["init_obj_axis_angle"].to(device)
            init_obj_translation = ckpt["init_obj_translation"].to(device)
        step_count = int(ckpt.get("step_count", 0))
        print(f"Resumed from {resume_from_checkpoint} (step {step_count}).")

    # Dump a "step 0" frame so you can visually confirm the starting state
    # (FP/HaMeR init or resumed checkpoint state) before optimization moves
    # things. Filename suffix marks resume to avoid overwriting the prior
    # run's 000000.png when extending training.
    if render_every > 0:
        start_step = step_count if resume_from_checkpoint is not None else 0
        _dump_progress_frame(
            progress_dir, start_step, debug_frame_t, cache,
            K, W, H, device,
            obj_gaussians, obj_pose_field, hand_slots,
            bg_gaussians=bg_gaussians, bg_pose_field=bg_pose_field,
        )

    # Pose-param requires_grad is toggled per-epoch by two independent
    # gates:
    #   - the Gaussian-only warmup (first N epochs): freezes everything.
    #   - explicit freeze_* user flags: hold for the entire run.
    # Both work via toggling requires_grad rather than excluding params
    # from the optimizer, so Adam's running moments aren't corrupted by
    # stale state when params are re-enabled at the warmup boundary.
    def _set_pose_grad_state(epoch: int) -> None:
        in_warmup = epoch < n_gaussian_only_epochs
        # Whole-Gaussian freeze: pins every Gaussian-attribute parameter
        # (obj + hand + bg) for the entire run. Combine with freeze_hand_*
        # and freeze_bg_* to do "object-pose-only" optimization while
        # everything else stays exactly as the (possibly resumed) state
        # left it.
        if freeze_gaussians:
            for p in obj_gaussians.parameters():
                p.requires_grad_(False)
            for slot in hand_slots:
                for p in slot.gaussians.parameters():
                    p.requires_grad_(False)
            if bg_gaussians is not None:
                for p in bg_gaussians.parameters():
                    p.requires_grad_(False)
        obj_pose_field.axis_angle.requires_grad_(
            not in_warmup and not freeze_object_rot)
        obj_pose_field.translation.requires_grad_(
            not in_warmup and not freeze_object_trans)
        # Global object scale: lives on ObjectGaussians, not pose field;
        # toggle the same way to keep behavior consistent across the run.
        obj_gaussians._log_scale_global.requires_grad_(
            not in_warmup and not freeze_object_scale)
        for slot in hand_slots:
            slot.pose_field.global_orient.requires_grad_(
                not in_warmup and not freeze_hand_rot)
            slot.pose_field.hand_pose.requires_grad_(
                not in_warmup and not freeze_hand_rot)
            slot.pose_field.cam_t.requires_grad_(
                not in_warmup and not freeze_hand_trans)
            # Betas only frozen during warmup; no user-level freeze knob.
            slot.pose_field.betas.requires_grad_(not in_warmup)
            # hand_scale gated on both --learn_hand_scale (user opt-in) AND
            # warmup. Outside warmup with the flag set, the prior loss
            # also activates.
            slot.pose_field.hand_scale.requires_grad_(
                learn_hand_scale and not in_warmup)
        if bg_pose_field is not None:
            bg_pose_field.axis_angle.requires_grad_(
                not in_warmup and not freeze_bg_rot)
            bg_pose_field.translation.requires_grad_(
                not in_warmup and not freeze_bg_trans)

    if any([freeze_object_rot, freeze_object_trans,
            freeze_hand_rot,   freeze_hand_trans]):
        flags = []
        if freeze_object_rot:   flags.append("object rot")
        if freeze_object_trans: flags.append("object trans")
        if freeze_hand_rot:     flags.append("hand rot")
        if freeze_hand_trans:   flags.append("hand trans")
        print(f"Frozen for the entire run: {', '.join(flags)}")

    for epoch in range(n_epochs):
        _set_pose_grad_state(epoch)
        if epoch == 0 and n_gaussian_only_epochs > 0:
            print(f"Epochs 0..{n_gaussian_only_epochs - 1}: all poses "
                  f"frozen (Gaussian-only warmup).")
        if epoch == n_gaussian_only_epochs:
            print(f"Epoch {epoch}: warmup over; pose optimization on "
                  f"(per freeze_* flags).")

        # Discrete rotation search: runs at the warmup boundary (when
        # poses unfreeze for the first time, Gaussians are settled enough
        # to score candidates well) and optionally every K epochs after.
        if rotation_search_n_candidates > 0 and not freeze_object_rot:
            do_search = (epoch == n_gaussian_only_epochs)
            if rotation_search_period > 0 and epoch >= n_gaussian_only_epochs:
                do_search = do_search or (
                    (epoch - n_gaussian_only_epochs) % rotation_search_period == 0
                )
            if do_search:
                pbar.set_description(f"refine (rot search at epoch {epoch})")
                n_upd = _discrete_rotation_search(
                    obj_gaussians          = obj_gaussians,
                    obj_pose_field         = obj_pose_field,
                    optimizer              = optimizer,
                    cache                  = cache,
                    K=K, W=W, H=H, device=device,
                    n_candidates           = rotation_search_n_candidates,
                    local_frac             = rotation_search_local_frac,
                    local_max_deg          = rotation_search_local_max_deg,
                    silhouette_weight      = rotation_search_silhouette_weight,
                    smoothness_weight      = rotation_search_smoothness_weight,
                    seed                   = seed + epoch,
                )
                print(f"Rotation search updated {n_upd}/{n_frames} frames.")
                pbar.set_description("refine")

        order = rng.permutation(n_frames)              # frame visit order
        for batch_start in range(0, n_frames, batch_size):
            batch_ts = order[batch_start : batch_start + batch_size]

            optimizer.zero_grad(set_to_none=True)
            total = 0.0

            # Coarse-to-fine multiplier on Gaussian render-scales for this batch.
            # Constant within the batch so all sampled frames see the same blur.
            if coarse_init_scale_factor != 1.0:
                anneal_progress = min(
                    step_count / max(1, coarse_decay_steps - 1), 1.0
                )
                coarse_scale_mul = math.exp(
                    math.log(coarse_init_scale_factor) * (1.0 - anneal_progress)
                )
            else:
                coarse_scale_mul = 1.0

            # Dynamic per-frame confidence (recomputed each batch from
            # current pose state). Multiplies with the static confidence.
            if pose_confidence_dynamic_tau > 0.0:
                pose_confidence_dyn = _compute_dynamic_pose_confidence()
            else:
                pose_confidence_dyn = None

            # ---- batched per-frame compute (cache + poses) ----------------
            # Single-pass index tensor for the batch. All per-frame quantities
            # that don't depend on Gaussian rasterization (MANO LBS, object
            # pose, bg pose, cache fetches) are computed in one big tensor op
            # here, instead of B small ops in the per-frame loop below. The
            # ManoLayer call alone goes from ~B kernel-launch-bound batch=1
            # invocations to a single batch=B invocation — biggest single win.
            batch_ts_t = torch.as_tensor(
                [int(x) for x in batch_ts], device=device, dtype=torch.long
            )
            B = int(batch_ts_t.shape[0])

            # Cache fetch — single H2D transfer per buffer instead of B.
            batch_rgb       = cache.rgb[batch_ts_t.cpu()].to(device, non_blocking=True)         # (B, H, W, 3)
            batch_obj_mask  = cache.obj_mask[batch_ts_t.cpu()].to(device, non_blocking=True)    # (B, H, W)
            batch_hand_msks = [
                m[batch_ts_t.cpu()].to(device, non_blocking=True) for m in cache.hand_masks
            ]                                                                                    # list of (B, H, W)
            batch_depth = (
                cache.depth[batch_ts_t.cpu()].to(device, non_blocking=True)
                if cache.depth is not None else None
            )                                                                                    # (B, H, W) or None
            # Union mask per batch member (vectorized).
            batch_union = batch_obj_mask.clone()
            for hm in batch_hand_msks:
                batch_union = torch.maximum(batch_union, hm)

            # Poses, batched.
            R_obj_b, t_obj_b = obj_pose_field.batched_forward(batch_ts_t)                       # (B, 3, 3), (B, 3)
            hand_batched: list[tuple[torch.Tensor, torch.Tensor]] = []
            for slot in hand_slots:
                hv, hr = slot.pose_field.batched_posed_verts_and_rotmats_camera(batch_ts_t)
                hand_batched.append((hv, hr))                                                    # (B, V, 3), (B, V, 3, 3)
            if bg_gaussians is not None:
                R_bg_b, t_bg_b = bg_pose_field.batched_forward(batch_ts_t)                      # (B, 3, 3), (B, 3)

            # Per-frame losses, summed across the batch then backward once.
            for i, t in enumerate(batch_ts):
                t = int(t)
                obj_mask  = batch_obj_mask[i]
                hand_msks = [hm[i] for hm in batch_hand_msks]
                union_mask = batch_union[i]
                target_rgb_t = batch_rgb[i]

                obj_frame   = obj_gaussians(R_obj_b[i], t_obj_b[i])
                hand_frames = [
                    slot.gaussians(hv[i], hr[i])
                    for slot, (hv, hr) in zip(hand_slots, hand_batched)
                ]
                bg_frame = None
                if bg_gaussians is not None:
                    bg_frame = bg_gaussians(R_bg_b[i], t_bg_b[i])

                all_frames = [obj_frame] + hand_frames
                if bg_frame is not None:
                    all_frames.append(bg_frame)
                combined = concat_frames(all_frames)
                if coarse_scale_mul != 1.0:
                    combined = dataclasses.replace(
                        combined, scales=combined.scales * coarse_scale_mul
                    )

                # Class labels are static (depend only on Gaussian counts);
                # built once at startup as ``labels_static``.
                rgb_pred, depth_pred, _, class_pred = render_rgb_depth(
                    combined, K, W, H, extra_features=labels_static)

                # Photometric:
                #   - balance_photometric_by_mask: per-entity inverse-area
                #     weighting so each entity contributes equally.
                #   - else if bg_gaussians: full-image L1.
                #   - else: foreground-only L1 (or black-bg masked variant).
                if balance_photometric_by_mask:
                    fl = weights.photometric * balanced_photometric_loss(
                        rgb_pred, target_rgb_t,
                        obj_mask, hand_msks,
                        include_background=(bg_gaussians is not None),
                        use_l2=use_l2_photometric,
                    )
                elif bg_gaussians is not None:
                    diff_full = rgb_pred - target_rgb_t
                    if use_l2_photometric:
                        fl = weights.photometric * (diff_full * diff_full).mean()
                    else:
                        fl = weights.photometric * diff_full.abs().mean()
                else:
                    fl = weights.photometric * photometric_loss(
                        rgb_pred, target_rgb_t, union_mask,
                        mask_background_to_black=mask_background_to_black,
                        use_l2=use_l2_photometric,
                    )

                # Class-label silhouette target.
                target_class = obj_mask.new_zeros((H, W, K_classes_static))
                target_class[..., 0] = obj_mask
                for k_h, hmask in enumerate(hand_msks):
                    target_class[..., k_h + 1] = hmask
                fl = fl + weights.silhouette * silhouette_loss(
                    class_pred, target_class,
                    class_weights=silhouette_class_weights,
                    use_l2=use_l2_silhouette,
                )

                if batch_depth is not None:
                    fl = fl + weights.depth * depth_loss(
                        depth_pred, batch_depth[i], union_mask)

                # Confidence weighting (static × dynamic).
                c_t = pose_confidence[t]
                if pose_confidence_dyn is not None:
                    c_t = c_t * pose_confidence_dyn[t]
                fl = fl * c_t

                total = total + fl

            # Average per-frame losses so loss magnitude is independent of B.
            loss = total / float(len(batch_ts))

            # Sequence-wide regularizers (computed once per step, not per-frame).
            loss = loss + weights.delta_p_reg * delta_p_regularizer(
                obj_gaussians._delta_p
            )
            # Rotations get quat-aligned smoothness so axis-angle
            # double-cover wraps don't induce phantom large differences.
            loss = loss + (
                weights.smooth_obj_rot   * rotation_smoothness(obj_pose_field.axis_angle) +
                weights.smooth_obj_trans * temporal_smoothness(obj_pose_field.translation)
            )
            # Tight prior on global object scale (log = 0 → s = 1).
            loss = loss + weights.obj_scale_prior * (
                obj_gaussians._log_scale_global ** 2
            )
            # Per-frame pose-init prior weighted by confidence: pulls
            # high-confidence frames toward FoundationPose's input pose,
            # leaves low-confidence frames free.
            if w_pose_init_prior > 0.0:
                loss = loss + w_pose_init_prior * pose_init_prior_loss(
                    obj_pose_field.axis_angle,
                    obj_pose_field.translation,
                    init_obj_axis_angle,
                    init_obj_translation,
                    pose_confidence,
                )
            if bg_gaussians is not None:
                loss = loss + weights.delta_p_reg * delta_p_regularizer(
                    bg_gaussians._delta_p
                )
                loss = loss + (
                    weights.smooth_bg_rot   * rotation_smoothness(bg_pose_field.axis_angle) +
                    weights.smooth_bg_trans * temporal_smoothness(bg_pose_field.translation)
                )
            for slot in hand_slots:
                loss = loss + weights.delta_p_reg * delta_p_regularizer(
                    slot.gaussians._delta_p
                )
                T_h = slot.pose_field.hand_pose.shape[0]
                loss = loss + (
                    weights.smooth_hand_rot    * rotation_smoothness(slot.pose_field.global_orient) +
                    weights.smooth_hand_finger * rotation_smoothness(slot.pose_field.hand_pose.view(T_h, 15, 3)) +
                    weights.smooth_hand_trans  * temporal_smoothness(slot.pose_field.cam_t)
                )
                loss = loss + weights.beta_prior * beta_prior_loss(
                    slot.pose_field.betas, slot.pose_field.betas_init
                )
                if slot.pose_field.hand_scale.requires_grad:
                    loss = loss + weights.hand_scale_prior * hand_scale_prior_loss(
                        slot.pose_field.hand_scale, slot.pose_field.hand_scale_init
                    )

            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            step_count += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{float(loss):.4f}", epoch=epoch)

            if render_every > 0 and step_count % render_every == 0:
                _dump_progress_frame(
                    progress_dir, step_count, debug_frame_t, cache,
                    K, W, H, device,
                    obj_gaussians, obj_pose_field, hand_slots,
                    bg_gaussians=bg_gaussians, bg_pose_field=bg_pose_field,
                )

    pbar.close()

    # Stitch progress frames into a video for easy review.
    if render_every > 0 and progress_dir is not None:
        _stitch_progress_video(progress_dir)

    # ----- save outputs -----------------------------------------------------
    s_obj_learned = float(obj_gaussians.object_scale().detach())
    print(f"Learned object scale: {s_obj_learned:.4f}")
    refined_obj_track = obj_pose_field.export_track()
    if refined_object_scale_path is None:
        # Legacy path: bake the learned global scale into the per-frame
        # Transform3d.scale so downstream renderers that read pd["scale"]
        # see the correct size without needing to know about s_obj.
        refined_obj_track.scales = refined_obj_track.scales * s_obj_learned
    else:
        # Clean path: keep per-frame scales as input, save the learned
        # scale to its own JSON for explicit consumption by the renderer.
        os.makedirs(
            os.path.dirname(os.path.abspath(refined_object_scale_path)) or ".",
            exist_ok=True,
        )
        import json as _json
        with open(refined_object_scale_path, "w") as f:
            _json.dump({"scale": s_obj_learned}, f, indent=2)
        print(f"Wrote learned object scale → {refined_object_scale_path}")
    save_object_poses(refined_obj_track, refined_object_poses_dir)
    print(f"Wrote refined object poses → {refined_object_poses_dir}")
    for slot in hand_slots:
        if slot.output_pose_dir is None:
            continue
        save_hand_poses(slot.pose_field.export_track(slot.track.raw_records),
                        slot.output_pose_dir)
        print(f"Wrote refined {slot.side}-hand poses → {slot.output_pose_dir}")

    # ----- optional checkpoint save ----------------------------------------
    if checkpoint_path is not None:
        os.makedirs(
            os.path.dirname(os.path.abspath(checkpoint_path)) or ".",
            exist_ok=True,
        )
        ckpt_save = {
            "obj_gaussians":        obj_gaussians.state_dict(),
            "obj_pose_field":       obj_pose_field.state_dict(),
            "hand_gaussians":       [s.gaussians.state_dict()  for s in hand_slots],
            "hand_pose_fields":     [s.pose_field.state_dict() for s in hand_slots],
            "hand_sides":           [s.side for s in hand_slots],
            "optimizer":            optimizer.state_dict(),
            "init_obj_axis_angle":  init_obj_axis_angle.detach().cpu(),
            "init_obj_translation": init_obj_translation.detach().cpu(),
            "step_count":           step_count,
            "frame_indices":        list(frame_indices),
            "s_obj_learned":        s_obj_learned,
        }
        if bg_gaussians is not None:
            ckpt_save["bg_gaussians"]  = bg_gaussians.state_dict()
            ckpt_save["bg_pose_field"] = bg_pose_field.state_dict()
        if lr_scheduler is not None:
            ckpt_save["lr_scheduler"]  = lr_scheduler.state_dict()
        # Labeled snapshot of Adam's per-frame squared-gradient EMA on
        # pose params. The full optimizer state is also in ``optimizer``
        # above (by integer index), but extracting by name is much easier
        # for downstream plotting.
        def _exp_avg_sq(p):
            v = optimizer.state.get(p, {}).get("exp_avg_sq")
            return v.detach().cpu() if v is not None else None
        ckpt_save["pose_grad_state"] = {
            "obj_axis_angle":  _exp_avg_sq(obj_pose_field.axis_angle),
            "obj_translation": _exp_avg_sq(obj_pose_field.translation),
            "hands": [
                {
                    "side":           slot.side,
                    "global_orient":  _exp_avg_sq(slot.pose_field.global_orient),
                    "hand_pose":      _exp_avg_sq(slot.pose_field.hand_pose),
                    "cam_t":          _exp_avg_sq(slot.pose_field.cam_t),
                }
                for slot in hand_slots
            ],
        }
        torch.save(ckpt_save, checkpoint_path)
        print(f"Wrote checkpoint → {checkpoint_path}")

    # Final overlay video — streamed render + ffmpeg encode in one pass.
    _render_overlay_video_streaming(
        cache, frame_indices, K, W, H, device,
        obj_gaussians, obj_pose_field, hand_slots,
        output_path=overlay_path,
        bg_gaussians=bg_gaussians, bg_pose_field=bg_pose_field,
        include_background=multiview_include_background,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_frame_path(frames_dir: str, fidx: int) -> str:
    for ext in (".png", ".jpg"):
        p = os.path.join(frames_dir, f"{fidx:06d}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Frame {fidx:06d} not found in {frames_dir}")


def _dump_progress_frame(
    progress_dir: str,
    step: int,
    t: int,
    cache,
    K, W: int, H: int, device,
    obj_gaussians, obj_pose_field, hand_slots,
    bg_gaussians=None, bg_pose_field=None,
) -> None:
    """Render a single reference frame's overlay and save it as a PNG.

    Cheap: one combined gsplat rasterization, no per-set passes, no disk
    decoding (RGB is read straight from the in-memory cache).
    """
    with torch.no_grad():
        rgb = cache.rgb[t].to(device, non_blocking=True)
        R_obj, t_obj = obj_pose_field(t)
        obj_frame    = obj_gaussians(R_obj, t_obj)
        hand_frames  = []
        for s in hand_slots:
            v, R = s.pose_field.posed_verts_and_rotmats_camera(t)
            hand_frames.append(s.gaussians(v, R))
        all_frames = [obj_frame] + hand_frames
        if bg_gaussians is not None:
            R_bg, t_bg = bg_pose_field(t)
            all_frames.append(bg_gaussians(R_bg, t_bg))
        combined     = concat_frames(all_frames)
        rgb_pred, _, alpha, _ = render_rgb_depth(combined, K, W, H)
        # When background is rendered, alpha covers the whole image — show
        # the rendered scene directly. Otherwise overlay foreground onto image.
        if bg_gaussians is not None:
            mix = rgb_pred
        else:
            mix = rgb * (1.0 - alpha.unsqueeze(-1)) + rgb_pred * alpha.unsqueeze(-1)
        arr = (mix.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(arr).save(os.path.join(progress_dir, f"{step:06d}.png"))


def _stitch_progress_video(progress_dir: str, fps: float = 30.0) -> None:
    """Stitch progress PNGs into ``progress_dir/../progress.mp4``."""
    pngs = sorted(os.listdir(progress_dir))
    pngs = [p for p in pngs if p.endswith(".png")]
    if not pngs:
        return
    out_path = os.path.join(os.path.dirname(progress_dir.rstrip("/")) or ".",
                            "progress.mp4")
    # ffmpeg's -pattern_type glob handles non-contiguous step numbers cleanly.
    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-pattern_type", "glob",
            "-i", os.path.join(progress_dir, "*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            out_path,
        ], check=True)
        print(f"Wrote progress video → {out_path}")
    except subprocess.CalledProcessError:
        print(f"Warning: failed to stitch progress video from {progress_dir}")


def _generate_rotation_candidates(
    current_rotmat: torch.Tensor,    # (3, 3)
    n_candidates: int,
    n_local: int,
    local_max_rad: float,
    seed: int,
) -> torch.Tensor:
    """Return (n_candidates, 3, 3) rotation matrices.

    Layout:
      idx 0           — current rotation (so the search never regresses)
      idx 1..n_local  — local perturbations of current (axis-angle within
                        ±local_max_rad on a random axis)
      idx n_local+1.. — global uniform-on-SO(3) rotations
    """
    from .gaussians import axis_angle_to_quat, quat_to_rotmat

    device = current_rotmat.device
    rots = [current_rotmat]
    g = torch.Generator(device="cpu").manual_seed(seed)

    if n_local > 0:
        axes = torch.randn(n_local, 3, generator=g)
        axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        angles = torch.rand(n_local, generator=g) * local_max_rad
        aa = (axes * angles.unsqueeze(-1)).to(device)
        q_perturb = axis_angle_to_quat(aa)                 # (n_local, 4)
        R_perturb = quat_to_rotmat(q_perturb)              # (n_local, 3, 3)
        # Apply perturbation in current frame: R = R_perturb @ R_current.
        rots.append((R_perturb @ current_rotmat.unsqueeze(0)).reshape(-1, 3, 3))

    n_global = n_candidates - 1 - n_local
    if n_global > 0:
        # Uniform on SO(3) via random unit quaternions.
        q_rand = torch.randn(n_global, 4, generator=g)
        q_rand = q_rand / q_rand.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # Convert (x,y,z,w) order doesn't matter here since we feed straight
        # to quat_to_rotmat which expects (w,x,y,z) — randn samples are
        # rotation-isotropic either way.
        R_global = quat_to_rotmat(q_rand.to(device))       # (n_global, 3, 3)
        rots.append(R_global)

    out = torch.cat([r if r.dim() == 3 else r.unsqueeze(0) for r in rots], dim=0)
    return out


def _discrete_rotation_search(
    obj_gaussians,
    obj_pose_field,
    optimizer,
    cache,
    K, W: int, H: int, device,
    n_candidates: int,
    local_frac: float,
    local_max_deg: float,
    silhouette_weight: float,
    smoothness_weight: float,
    seed: int,
) -> int:
    """Per-frame batched rotation hypothesis search for the object pose.

    For each frame, render all N candidate rotations in a *single* gsplat
    call (each candidate is a different viewmat sharing the same Gaussian
    set in canonical frame), score by photometric L1 + silhouette IoU
    against the SAM2 target, and snap the frame's axis_angle to the best
    candidate. Optimizer state for the axis_angle parameter is cleared so
    Adam's running moments don't fight the snap.

    Returns the number of frames whose pose was updated.
    """
    from .gaussians import axis_angle_to_quat, quat_to_rotmat, rotmat_to_quat
    from .pose_fields import _quat_to_axis_angle
    from gsplat.rendering import rasterization

    n_frames = obj_pose_field.axis_angle.shape[0]
    n_local  = max(1, int(local_frac * (n_candidates - 1)))
    n_local  = min(n_local, n_candidates - 1)

    # Canonical Gaussians (object frame, with global object scale baked in).
    # Calling forward(I, 0) gives positions in object frame.
    with torch.no_grad():
        canon = obj_gaussians(
            torch.eye(3, device=device),
            torch.zeros(3, device=device),
        )

    n_updated = 0
    pbar = tqdm(range(n_frames), ncols=80, desc="rotation search")
    with torch.no_grad():
        for t in pbar:
            current_aa = obj_pose_field.axis_angle[t]
            current_rot = quat_to_rotmat(axis_angle_to_quat(current_aa))   # (3, 3)
            current_t   = obj_pose_field.translation[t]                    # (3,)

            candidates = _generate_rotation_candidates(
                current_rotmat=current_rot,
                n_candidates=n_candidates,
                n_local=n_local,
                local_max_rad=float(local_max_deg) * 3.141592653589793 / 180.0,
                seed=seed + t,
            )                                                              # (N, 3, 3)
            N = candidates.shape[0]

            # Build (N, 4, 4) viewmats with same translation, varying rotation.
            viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
            viewmats[:, :3, :3] = candidates
            viewmats[:, :3,  3] = current_t

            Ks = K.unsqueeze(0).expand(N, 3, 3).contiguous()
            out, alphas, _ = rasterization(
                means     = canon.means,
                quats     = canon.quats,
                scales    = canon.scales,
                opacities = canon.opacities,
                colors    = canon.colors,
                viewmats  = viewmats,
                Ks        = Ks,
                width     = W,
                height    = H,
                render_mode = "RGB",
            )
            # out: (N, H, W, 3); alphas: (N, H, W, 1)

            target_rgb  = cache.rgb[t].to(device, non_blocking=True)        # (H, W, 3)
            target_mask = cache.obj_mask[t].to(device, non_blocking=True)   # (H, W)

            photo = (out - target_rgb.unsqueeze(0)).abs().mean(dim=(1, 2, 3))  # (N,)
            alpha = alphas[..., 0]                                           # (N, H, W)
            inter = (alpha * target_mask.unsqueeze(0)).sum(dim=(1, 2))
            denom = (alpha.sum(dim=(1, 2)) + target_mask.sum() - inter).clamp_min(1e-6)
            iou   = inter / denom

            score = -photo + silhouette_weight * iou                       # (N,)

            # Causal smoothness: penalize candidates that diverge from the
            # previous (already-snapped) frame's rotation. Forward sweep is
            # implicit in the for-t loop: by the time we score frame t,
            # frame t-1 has its final snapped axis_angle. Quat-aligned
            # squared distance handles the q ≡ -q double cover.
            if t > 0 and smoothness_weight > 0.0:
                q_cand = rotmat_to_quat(candidates)                         # (N, 4)
                q_prev = axis_angle_to_quat(
                    obj_pose_field.axis_angle[t - 1].unsqueeze(0)
                )[0]                                                        # (4,)
                dot = (q_cand * q_prev.unsqueeze(0)).sum(dim=-1, keepdim=True)
                q_prev_aligned = torch.where(
                    dot < 0,
                    -q_prev.unsqueeze(0).expand_as(q_cand),
                     q_prev.unsqueeze(0).expand_as(q_cand),
                )
                smooth_pen = ((q_cand - q_prev_aligned) ** 2).sum(dim=-1)   # (N,)
                score = score - smoothness_weight * smooth_pen

            best  = int(score.argmax())

            if best != 0:
                best_R = candidates[best]
                # rotmat → quat → axis-angle.
                best_q = rotmat_to_quat(best_R)
                best_aa = _quat_to_axis_angle(best_q.unsqueeze(0))[0]
                obj_pose_field.axis_angle.data[t] = best_aa
                n_updated += 1

    # Clear Adam state for the snapped parameter so running moments don't
    # fight the new init. State will be lazily re-initialized on the next
    # optimizer.step() call.
    if obj_pose_field.axis_angle in optimizer.state:
        optimizer.state[obj_pose_field.axis_angle] = {}

    return n_updated


def _render_overlay_sequence(
    frames_dir, frame_indices, K, W, H, device,
    obj_gaussians, obj_pose_field, hand_slots,
    bg_gaussians=None, bg_pose_field=None,
) -> list[np.ndarray]:
    """Render every frame as image * (1 - α) + render * α and return uint8 RGBs.

    Disk-reads source RGB per frame and accumulates into a list. Slow; kept
    only for completeness. Production path is
    ``_render_overlay_video_streaming`` which streams cache→render→ffmpeg
    without materializing the full sequence in memory.
    """
    out = []
    with torch.no_grad():
        for t, fidx in enumerate(frame_indices):
            rgb = load_rgb(_find_frame_path(frames_dir, fidx), str(device))
            R_obj, t_obj = obj_pose_field(t)
            obj_frame    = obj_gaussians(R_obj, t_obj)
            hand_frames  = []
            for s in hand_slots:
                v, R = s.pose_field.posed_verts_and_rotmats_camera(t)
                hand_frames.append(s.gaussians(v, R))
            all_frames = [obj_frame] + hand_frames
            if bg_gaussians is not None:
                R_bg, t_bg = bg_pose_field(t)
                all_frames.append(bg_gaussians(R_bg, t_bg))
            combined     = concat_frames(all_frames)
            rgb_pred, _, alpha, _ = render_rgb_depth(combined, K, W, H)
            if bg_gaussians is not None:
                mix = rgb_pred
            else:
                mix = rgb * (1.0 - alpha.unsqueeze(-1)) + rgb_pred * alpha.unsqueeze(-1)
            out.append((mix.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
    return out


def _orbit_viewmat_cv(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """world→camera (4, 4) in CV convention (cam +Z forward, +Y down)."""
    fwd = target - eye
    fwd = fwd / max(np.linalg.norm(fwd), 1e-8)
    x = np.cross(up, fwd)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        # up ∥ fwd: pick a fallback up perpendicular to fwd.
        alt = np.array([1.0, 0.0, 0.0]) if abs(fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(alt, fwd)
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(fwd, x)
    R = np.stack([x, y, fwd], axis=0)                     # rows = world→cam basis
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3]  = -R @ eye
    return M


def _render_overlay_video_streaming(
    cache, frame_indices, K, W, H, device,
    obj_gaussians, obj_pose_field, hand_slots,
    output_path: str,
    bg_gaussians=None, bg_pose_field=None,
    fps: float = 30.0,
    include_background: bool = False,
) -> None:
    """Multi-viewpoint overlay video, streamed render+ffmpeg in one pass.

    Per frame we render four panels in a single batched gsplat call (one
    rasterization, four viewmats):

        [ src cam (image underlay) ]   [ world top  ]
        [ world side               ]   [ world back ]

    World cameras orbit the *current frame's* object centroid, so each
    frame is independently centered (matches the mesh-render grid in
    v2d_hamer/render_hands_aligned_video.py — "world" is just whatever
    coordinate frame the Gaussians live in for that frame, which is the
    camera frame at time t).

    Cost: one rasterization at 4× viewmats per frame (gsplat handles the
    4-camera batch natively with shared Gaussians), plus PIL grid stitch.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # Each panel is half resolution; the full 2x2 grid output is W × H.
    pw = max(2, (W // 2) & ~1)        # even, ≥ 2
    ph = max(2, (H // 2) & ~1)
    grid_W = 2 * pw
    grid_H = 2 * ph

    # Intrinsics for half-resolution panels.
    K_panel = K.clone()
    K_panel[0, 0] *= pw / W
    K_panel[1, 1] *= ph / H
    K_panel[0, 2] *= pw / W
    K_panel[1, 2] *= ph / H

    from gsplat.rendering import rasterization
    panel_labels = ["src cam", "top", "side", "behind"]

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{grid_W}x{grid_H}",
            "-framerate", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )

    pbar = tqdm(total=len(frame_indices), ncols=80, desc="overlay video")
    try:
        with torch.no_grad():
            bg_zeros = torch.zeros(4, 3, device=device, dtype=torch.float32)
            for t, _fidx in enumerate(frame_indices):
                # Source image at panel resolution.
                src_rgb_full = cache.rgb[t].to(device, non_blocking=True)         # (H, W, 3)
                src_rgb_panel = torch.nn.functional.interpolate(
                    src_rgb_full.permute(2, 0, 1).unsqueeze(0),
                    size=(ph, pw), mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0).contiguous()                        # (ph, pw, 3)

                # Build per-frame Gaussian set in current cam frame.
                # By default we exclude the background Gaussians so the
                # orbit panels show only the trained foreground (object +
                # hands) — easier to read shape/pose quality. The src-cam
                # panel uses the same foreground-only render and gets
                # composited over the source image, recovering the
                # "scene with foreground overlay" look.
                R_obj, t_obj = obj_pose_field(t)
                obj_frame    = obj_gaussians(R_obj, t_obj)
                hand_frames  = []
                for s in hand_slots:
                    v, R = s.pose_field.posed_verts_and_rotmats_camera(t)
                    hand_frames.append(s.gaussians(v, R))
                all_frames = [obj_frame] + hand_frames
                if include_background and bg_gaussians is not None:
                    R_bg, t_bg = bg_pose_field(t)
                    all_frames.append(bg_gaussians(R_bg, t_bg))
                combined = concat_frames(all_frames)

                # Object centroid + radius in current cam frame (used for orbit).
                obj_means = obj_frame.means
                centroid_t = obj_means.mean(dim=0)
                radius_t   = (obj_means - centroid_t).norm(dim=-1).max().clamp_min(0.05)
                centroid = centroid_t.cpu().numpy()
                r = float(radius_t) * 2.5

                # Four viewmats:
                #   0: src cam (identity — Gaussians already in cam frame).
                #   1: top    — eye above (-Y), looking down; image-up = +Z (forward).
                #   2: side   — eye to the right (+X), looking inward.
                #   3: behind — eye in +Z (further than centroid), looking back.
                viewmats_np = np.stack([
                    np.eye(4),
                    _orbit_viewmat_cv(centroid + np.array([0.0, -r, 0.0]),
                                      centroid, up=np.array([0.0, 0.0, 1.0])),
                    _orbit_viewmat_cv(centroid + np.array([ r,  0.0, 0.0]),
                                      centroid, up=np.array([0.0, -1.0, 0.0])),
                    _orbit_viewmat_cv(centroid + np.array([0.0, 0.0,  r]),
                                      centroid, up=np.array([0.0, -1.0, 0.0])),
                ], axis=0)
                viewmats = torch.from_numpy(viewmats_np).to(device, dtype=torch.float32)
                Ks_4 = K_panel.unsqueeze(0).expand(4, 3, 3).contiguous()

                out, alphas, _ = rasterization(
                    means=combined.means,
                    quats=combined.quats,
                    scales=combined.scales,
                    opacities=combined.opacities,
                    colors=combined.colors,
                    viewmats=viewmats,
                    Ks=Ks_4,
                    width=pw, height=ph,
                    near_plane=0.01,
                    far_plane=100.0,
                    render_mode="RGB",
                    backgrounds=bg_zeros,
                )
                rgb_pred = out[..., :3].clamp(0, 1)                              # (4, ph, pw, 3)
                a0       = alphas[0, ..., 0].unsqueeze(-1)                       # (ph, pw, 1)

                # Panel 0: blend over source image. Panels 1-3: just the render
                # (against black bg, or full scene if bg gaussians active).
                panel_0 = src_rgb_panel * (1.0 - a0) + rgb_pred[0] * a0

                grid = torch.zeros(grid_H, grid_W, 3, device=device)
                grid[:ph, :pw] = panel_0
                grid[:ph, pw:] = rgb_pred[1]
                grid[ph:, :pw] = rgb_pred[2]
                grid[ph:, pw:] = rgb_pred[3]

                arr = (grid * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                arr = np.ascontiguousarray(arr)

                # Panel labels via PIL (cheap — adds ~1 ms/frame).
                img = Image.fromarray(arr)
                draw = ImageDraw.Draw(img)
                font = _font_cached()
                for i, label in enumerate(panel_labels):
                    x0 = (i % 2) * pw + 6
                    y0 = (i // 2) * ph + 6
                    tw = draw.textlength(label, font=font)
                    draw.rectangle([x0, y0, x0 + tw + 8, y0 + 22], fill=(0, 0, 0))
                    draw.text((x0 + 4, y0 + 3), label, fill=(255, 255, 255), font=font)
                arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))

                proc.stdin.write(arr.tobytes())
                pbar.update(1)
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()
        pbar.close()
    print(f"Wrote multi-view overlay video → {output_path}")


def _font_cached():
    """Lazy small bitmap font for grid labels."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16,
        )
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames_dir",                  required=True)
    p.add_argument("--intrinsics_path",             required=True)
    p.add_argument("--object_mesh_path",            required=True)
    p.add_argument("--object_poses_dir",            required=True)
    p.add_argument("--object_mask_dir",             required=True)
    p.add_argument("--refined_object_poses_dir",    required=True)
    p.add_argument("--overlay_path",                required=True)
    p.add_argument("--refined_object_scale_path",   default=None,
                   help="Optional path to write the learned global object "
                        "scale as a JSON file ``{\"scale\": float}``. When "
                        "set, per-frame Transform3d.scale is left as input "
                        "(scale is exported separately for the renderer).")
    p.add_argument("--left_hand_pose_dir",          default=None)
    p.add_argument("--left_hand_mask_dir",          default=None)
    p.add_argument("--right_hand_pose_dir",         default=None)
    p.add_argument("--right_hand_mask_dir",         default=None)
    p.add_argument("--refined_left_hand_pose_dir",  default=None)
    p.add_argument("--refined_right_hand_pose_dir", default=None)
    p.add_argument("--depth_dir",                   default=None)
    p.add_argument("--mano_assets_root",            default=None)
    p.add_argument("--n_epochs",        type=int,   default=30,
                   help="Number of full passes over the sequence. Each "
                        "epoch visits every frame once in random order.")
    p.add_argument("--n_gaussian_only_epochs", type=int, default=5,
                   help="Number of initial epochs with pose params frozen "
                        "(only Gaussian attributes update). Lets appearance "
                        "settle before poses start moving.")
    p.add_argument("--batch_size",      type=int,   default=4,
                   help="Frames per optimizer step. Per-frame losses are "
                        "averaged; sequence-wide regularizers (smoothness, "
                        "beta prior) are applied once per step.")
    p.add_argument("--lr_gaussians",    type=float, default=1e-2,
                   help="LR for object Gaussian attributes.")
    p.add_argument("--lr_hand_gaussians", type=float, default=None,
                   help="LR for hand Gaussian attributes (defaults to "
                        "--lr_gaussians). Bump up to compensate for the "
                        "smaller image area covered by hands.")
    # Per-attribute LR multipliers — shared across object/hand/bg Gaussian
    # sets, multiplied by their per-set base LR. Defaults of 1.0 preserve
    # old single-LR behavior. Standard 3DGS-ish ratios (relative to base
    # LR of 1e-2): delta_p≈0.016, quat≈0.1, scale≈0.5, opacity≈5, color≈0.25.
    p.add_argument("--lr_mul_delta_p",  type=float, default=1.0,
                   help="LR multiplier for Gaussian Δp (positions).")
    p.add_argument("--lr_mul_quat",     type=float, default=1.0,
                   help="LR multiplier for Gaussian quaternion.")
    p.add_argument("--lr_mul_scale",    type=float, default=1.0,
                   help="LR multiplier for Gaussian per-axis log-scale.")
    p.add_argument("--lr_mul_opacity",  type=float, default=1.0,
                   help="LR multiplier for Gaussian opacity logit.")
    p.add_argument("--lr_mul_color",    type=float, default=1.0,
                   help="LR multiplier for Gaussian color.")
    p.add_argument("--lr_mul_obj_global_scale", type=float, default=1.0,
                   help="LR multiplier for the global object-scale scalar.")
    p.add_argument("--lr_object_pose",  type=float, default=1e-3,
                   help="Legacy lumped LR for object pose (axis_angle + "
                        "translation). Overridden by --lr_object_rot / "
                        "--lr_object_trans if those are set.")
    p.add_argument("--lr_object_rot",   type=float, default=None,
                   help="Object rotation LR; defaults to --lr_object_pose. "
                        "Bump (e.g. 3-5x trans) if rotation converges slowly.")
    p.add_argument("--lr_object_trans", type=float, default=None,
                   help="Object translation LR; defaults to --lr_object_pose.")
    p.add_argument("--lr_hand_pose",    type=float, default=1e-3,
                   help="Legacy lumped LR for hand pose; overridden by "
                        "--lr_hand_global_orient / --lr_hand_finger / "
                        "--lr_hand_trans if those are set.")
    p.add_argument("--lr_hand_global_orient", type=float, default=None,
                   help="Hand wrist/root rotation LR.")
    p.add_argument("--lr_hand_finger",        type=float, default=None,
                   help="Hand finger articulation LR (15 joint axis-angles). "
                        "Often kept lower than global_orient/trans because "
                        "fingers tend to oscillate under photometric pull.")
    p.add_argument("--lr_hand_trans",         type=float, default=None,
                   help="Hand cam_t LR.")
    p.add_argument("--lr_betas",        type=float, default=1e-4)
    p.add_argument("--render_every",    type=int,   default=0,
                   help="Every N optimizer steps, dump an overlay PNG of "
                        "the reference frame to --progress_dir. Stitched "
                        "into progress.mp4 at the end. 0 disables.")
    p.add_argument("--progress_dir",                default=None,
                   help="Directory for per-step progress PNGs. Defaults "
                        "to <dir(overlay_path)>/progress/.")
    p.add_argument("--debug_frame_idx", type=int,   default=None,
                   help="Reference frame index for progress dumps "
                        "(defaults to the first frame in the sequence).")
    p.add_argument("--w_photometric",   type=float, default=1.0)
    p.add_argument("--w_silhouette",    type=float, default=0.5)
    p.add_argument("--w_silhouette_obj",  type=float, default=1.0,
                   help="Per-class relative weight for the object mask.")
    p.add_argument("--w_silhouette_hand", type=float, default=1.0,
                   help="Per-class relative weight for hand masks. "
                        "Drop below 1.0 (e.g. 0.3) when SAM2 hand masks "
                        "are noisier than the object mask.")
    p.add_argument("--w_depth",         type=float, default=0.05)
    p.add_argument("--w_smooth_obj_rot",    type=float, default=0.01,
                   help="Smoothness weight for object rotation (axis-angle, "
                        "in quat space). Set to 0 to disable rotation "
                        "smoothness while keeping translation smoothness.")
    p.add_argument("--w_smooth_obj_trans",  type=float, default=0.01)
    p.add_argument("--w_smooth_hand_rot",    type=float, default=0.01,
                   help="Smoothness weight for hand global_orient.")
    p.add_argument("--w_smooth_hand_finger", type=float, default=0.001,
                   help="Smoothness weight for finger articulation "
                        "(hand_pose, 15 joints). Default is 10x smaller "
                        "than --w_smooth_hand_rot since each finger joint "
                        "carries much weaker photometric signal — keeping "
                        "this low lets fingers articulate freely.")
    p.add_argument("--w_smooth_hand_trans",  type=float, default=0.01)
    p.add_argument("--w_beta_prior",    type=float, default=10.0)
    p.add_argument("--learn_hand_scale", action="store_true",
                   help="Optimize the per-track hand_scale buffer (init from "
                        "align_hands' n_pixels-weighted median). Off by "
                        "default; the init is usually already good and "
                        "betas covers the same DoF.")
    p.add_argument("--lr_hand_scale",     type=float, default=1e-3,
                   help="LR for hand_scale when --learn_hand_scale is set.")
    p.add_argument("--w_hand_scale_prior", type=float, default=10.0,
                   help="Tight prior pulling hand_scale back to the "
                        "align_hands estimate (only active when "
                        "--learn_hand_scale).")
    p.add_argument("--w_delta_p_reg",   type=float, default=100.0,
                   help="L2 penalty on Gaussian Δp (object + hand); pulls "
                        "splats back to mesh / MANO-rest anchor.")
    p.add_argument("--w_obj_scale_prior", type=float, default=1.0,
                   help="Tight Gaussian prior on log(global object scale). "
                        "Pulls scale toward 1.0; raise to freeze, lower to "
                        "let scale move freely.")
    p.add_argument("--mask_background_to_black", action="store_true",
                   help="Zero target's background and compute photometric "
                        "L1 over all pixels. Penalizes Gaussian leaks "
                        "outside the mask but makes mask errors costly.")
    p.add_argument("--balance_photometric_by_mask", action="store_true",
                   help="Per-pixel weight = 1/N_entity so object, each "
                        "hand, and (when --with_background) background "
                        "contribute equally to the photometric loss "
                        "regardless of pixel count. Prevents the largest "
                        "region from dominating the loss.")
    p.add_argument("--freeze_object_rot",   action="store_true",
                   help="Hold object rotation at its input value for the "
                        "entire run.")
    p.add_argument("--freeze_object_trans", action="store_true",
                   help="Hold object translation at its input value.")
    p.add_argument("--freeze_object_scale", action="store_true",
                   help="Hold global object scale at 1.0.")
    p.add_argument("--freeze_hand_rot",     action="store_true",
                   help="Hold hand rotations (global_orient + hand_pose) "
                        "at their input values for the entire run.")
    p.add_argument("--freeze_hand_trans",   action="store_true",
                   help="Hold hand cam_t at its input value.")
    p.add_argument("--with_background",     action="store_true",
                   help="Add background Gaussians (initialized from MoGe "
                        "depth at the reference frame) and a per-frame "
                        "rigid background→camera pose. Switches photometric "
                        "to a full-image L1.")
    p.add_argument("--bg_ref_frame",        type=int, default=None,
                   help="Reference frame index for background init. "
                        "Defaults to --debug_frame_idx, then frame 0.")
    p.add_argument("--lr_bg_gaussians",     type=float, default=None,
                   help="LR for background Gaussian attributes "
                        "(defaults to --lr_gaussians).")
    p.add_argument("--lr_bg_pose",          type=float, default=1e-3,
                   help="Legacy lumped LR for background pose; overridden "
                        "by --lr_bg_rot / --lr_bg_trans if those are set.")
    p.add_argument("--lr_bg_rot",           type=float, default=None)
    p.add_argument("--lr_bg_trans",         type=float, default=None)
    p.add_argument("--bg_max_points",       type=int,   default=50000,
                   help="Subsample background point cloud to this size.")
    p.add_argument("--background_pose_init_dir", default=None,
                   help="Optional folder of per-frame `Transform3d` JSONs "
                        "(camera-to-world; DROID/COLMAP convention) used "
                        "to seed the background pose field. With this set, "
                        "non-reference frames start at the relative pose "
                        "implied by the VO/SfM solution instead of identity, "
                        "which is the right basin for moving/egocentric "
                        "cameras.")
    p.add_argument("--bg_init_stride", type=int, default=10,
                   help="When background_pose_init_dir is set, fuse the BG "
                        "point cloud from MoGe depth at every Nth frame "
                        "(in addition to ref). 1 or 0 forces single-frame "
                        "(ref-only) init.")
    p.add_argument("--bg_voxel_size", type=float, default=0.005,
                   help="Voxel size (metres) used to dedup the fused BG "
                        "point cloud before the final random subsample. "
                        "Smaller = denser; 0 disables voxel dedup.")
    p.add_argument("--n_obj_gaussians",     type=int,   default=None,
                   help="Resample the object mesh surface to exactly this "
                        "many Gaussians. Default uses the mesh's vertex "
                        "count; useful for high-poly meshes (decimate) "
                        "or low-poly (densify on the surface).")
    p.add_argument("--n_hand_gaussians",    type=int,   default=None,
                   help="Subsample the 778 MANO vertices down to this "
                        "many Gaussians per hand. Default uses all 778. "
                        "Values > 778 are clipped (supersampling would "
                        "require barycentric LBS).")
    p.add_argument("--use_cosine_lr_schedule", action="store_true",
                   help="Apply a cosine LR decay across all param groups "
                        "over the full run. Per-group LR ratios are "
                        "preserved (each group's initial LR is multiplied "
                        "by the same cosine factor).")
    p.add_argument("--cosine_lr_min_ratio", type=float, default=0.0,
                   help="Final LR as a fraction of initial LR when cosine "
                        "schedule is on (e.g. 0.1 = decays to 10%% of "
                        "initial; 0.0 decays all the way to zero).")
    p.add_argument("--coarse_init_scale_factor", type=float, default=1.0,
                   help="Multiplier applied to all Gaussian render-scales "
                        "at training start. Decays log-linearly to 1.0 "
                        "over --coarse_decay_epochs. >1.0 widens the "
                        "photometric basin so outlier-pose frames have "
                        "gradient signal to be pulled in. 1.0 disables.")
    p.add_argument("--coarse_decay_epochs", type=int, default=None,
                   help="Epochs over which to anneal the coarse scale "
                        "factor back to 1.0 (default: full n_epochs).")
    p.add_argument("--pose_confidence_decay", type=float, default=0.0,
                   help="Per-frame pose confidence τ (in frames) for "
                        "exp(-|t-ref|/τ) decay from the reference frame. "
                        "Scales per-frame photometric/silhouette/depth "
                        "and modulates the pose-init prior. 0 disables "
                        "(uniform confidence).")
    p.add_argument("--pose_confidence_ref_frame", type=int, default=None,
                   help="Reference frame index (frame_idx, not positional) "
                        "for confidence decay. Falls back to "
                        "--bg_ref_frame, then 0.")
    p.add_argument("--pose_confidence_dynamic_tau", type=float, default=0.0,
                   help="Dynamic per-frame confidence based on quat-aligned "
                        "distance to neighbor frames' rotations, recomputed "
                        "each batch. c_dyn[t] = exp(-mean_neighbor_dist² / τ). "
                        "Multiplies with the static distance-from-ref "
                        "confidence. Set --pose_confidence_decay 0 to use "
                        "ONLY the dynamic confidence. 0 disables.")
    p.add_argument("--w_pose_init_prior", type=float, default=0.0,
                   help="Weight of the c_t-scaled pose-init prior. Pulls "
                        "high-confidence object poses back toward "
                        "FoundationPose input. 0 disables.")
    p.add_argument("--rotation_search_n_candidates", type=int, default=0,
                   help="Per-frame discrete rotation hypothesis search. "
                        "Render N candidate rotations in one batched gsplat "
                        "call, snap the frame's axis_angle to the best. "
                        "0 disables. 32-64 is typical.")
    p.add_argument("--rotation_search_period", type=int, default=0,
                   help="If >0, run rotation search every K epochs (after "
                        "the warmup boundary). 0 = run only once at the "
                        "warmup boundary.")
    p.add_argument("--rotation_search_local_frac", type=float, default=0.5,
                   help="Fraction of search candidates that are local "
                        "perturbations of the current rotation; the rest "
                        "are global uniform-on-SO(3) rotations.")
    p.add_argument("--rotation_search_local_max_deg", type=float, default=30.0,
                   help="Max angle (deg) for local perturbations.")
    p.add_argument("--rotation_search_silhouette_weight", type=float, default=1.0,
                   help="Silhouette IoU weight in candidate scoring "
                        "(photometric L1 weight is fixed at 1.0).")
    p.add_argument("--rotation_search_smoothness_weight", type=float, default=1.0,
                   help="Causal smoothness weight in candidate scoring. "
                        "Each candidate is penalized by its quat-aligned "
                        "squared distance to the previous (already-snapped) "
                        "frame's rotation. 0 disables, biasing only toward "
                        "image fit. The frame at t=0 has no penalty.")
    p.add_argument("--use_l2_photometric", action="store_true",
                   help="Use squared error (L2) for the photometric loss "
                        "instead of L1. Smoother gradient near the optimum, "
                        "less robust to outliers.")
    p.add_argument("--use_l2_silhouette", action="store_true",
                   help="Use squared error (L2) for the class-label "
                        "silhouette loss instead of L1.")
    p.add_argument("--train_resolution_scale", type=float, default=1.0,
                   help="Render at this scale of native resolution. "
                        "Cache, intrinsics, and renders all use the "
                        "scaled dimensions. 0.5 ≈ 4x faster rendering "
                        "with negligible accuracy hit. 1.0 = native.")
    p.add_argument("--multiview_include_background", action="store_true",
                   help="Include background Gaussians in the final multi-"
                        "view overlay grid. Default off — orbit panels "
                        "show only the trained foreground (object + "
                        "hands) for cleaner shape/pose QA.")
    p.add_argument("--checkpoint_path", default=None,
                   help="If set, save a .pt with full state (Gaussians + "
                        "poses + optimizer + LR scheduler + step counter) "
                        "at the end of training.")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="If set, load a previously-saved checkpoint before "
                        "the training loop. Module sizes must match (same "
                        "n_obj_gaussians / n_hand_gaussians / "
                        "bg_max_points / inputs).")
    p.add_argument("--ignore_optimizer_state", action="store_true",
                   help="When resuming, skip the optimizer (and LR "
                        "scheduler) state load — Adam moments reinit "
                        "lazily on the next step. Useful when changing "
                        "LRs / weights between runs.")
    p.add_argument("--freeze_gaussians", action="store_true",
                   help="Freeze ALL Gaussian-attribute parameters (object "
                        "+ hands + background) for the entire run. "
                        "Combine with --freeze_hand_rot / --freeze_hand_trans "
                        "/ --freeze_bg_rot / --freeze_bg_trans to leave "
                        "object pose as the only thing that updates.")
    p.add_argument("--random_init_obj_pose", action="store_true",
                   help="Randomize the per-frame object pose at startup "
                        "(uniform SO(3) rotation, translation += "
                        "N(0, σ²)) before training begins.")
    p.add_argument("--random_init_obj_pose_trans_std", type=float, default=0.1,
                   help="Stddev (m) of the per-frame translation noise "
                        "added when --random_init_obj_pose is set.")
    p.add_argument("--freeze_bg_rot",       action="store_true",
                   help="Hold background rotation at identity.")
    p.add_argument("--freeze_bg_trans",     action="store_true",
                   help="Hold background translation at zero.")
    p.add_argument("--w_smooth_bg_rot",     type=float, default=0.1)
    p.add_argument("--w_smooth_bg_trans",   type=float, default=0.1)
    p.add_argument("--device",                      default="cuda")
    p.add_argument("--seed",            type=int,   default=0)
    args = p.parse_args()
    refine(
        frames_dir                  = args.frames_dir,
        intrinsics_path             = args.intrinsics_path,
        object_mesh_path            = args.object_mesh_path,
        object_poses_dir            = args.object_poses_dir,
        object_mask_dir             = args.object_mask_dir,
        refined_object_poses_dir    = args.refined_object_poses_dir,
        refined_object_scale_path   = args.refined_object_scale_path,
        overlay_path                = args.overlay_path,
        left_hand_pose_dir          = args.left_hand_pose_dir,
        left_hand_mask_dir          = args.left_hand_mask_dir,
        right_hand_pose_dir         = args.right_hand_pose_dir,
        right_hand_mask_dir         = args.right_hand_mask_dir,
        refined_left_hand_pose_dir  = args.refined_left_hand_pose_dir,
        refined_right_hand_pose_dir = args.refined_right_hand_pose_dir,
        depth_dir                   = args.depth_dir,
        mano_assets_root            = args.mano_assets_root,
        n_epochs                    = args.n_epochs,
        n_gaussian_only_epochs      = args.n_gaussian_only_epochs,
        batch_size                  = args.batch_size,
        lr_gaussians                = args.lr_gaussians,
        lr_hand_gaussians           = args.lr_hand_gaussians,
        lr_mul_delta_p              = args.lr_mul_delta_p,
        lr_mul_quat                 = args.lr_mul_quat,
        lr_mul_scale                = args.lr_mul_scale,
        lr_mul_opacity              = args.lr_mul_opacity,
        lr_mul_color                = args.lr_mul_color,
        lr_mul_obj_global_scale     = args.lr_mul_obj_global_scale,
        lr_object_pose              = args.lr_object_pose,
        lr_object_rot               = args.lr_object_rot,
        lr_object_trans             = args.lr_object_trans,
        lr_hand_pose                = args.lr_hand_pose,
        lr_hand_global_orient       = args.lr_hand_global_orient,
        lr_hand_finger              = args.lr_hand_finger,
        lr_hand_trans               = args.lr_hand_trans,
        lr_betas                    = args.lr_betas,
        learn_hand_scale            = args.learn_hand_scale,
        lr_hand_scale               = args.lr_hand_scale,
        render_every                = args.render_every,
        progress_dir                = args.progress_dir,
        debug_frame_idx             = args.debug_frame_idx,
        mask_background_to_black    = args.mask_background_to_black,
        balance_photometric_by_mask = args.balance_photometric_by_mask,
        freeze_object_rot           = args.freeze_object_rot,
        freeze_object_trans         = args.freeze_object_trans,
        freeze_object_scale         = args.freeze_object_scale,
        freeze_hand_rot             = args.freeze_hand_rot,
        freeze_hand_trans           = args.freeze_hand_trans,
        with_background             = args.with_background,
        bg_ref_frame                = args.bg_ref_frame,
        lr_bg_gaussians             = args.lr_bg_gaussians,
        lr_bg_pose                  = args.lr_bg_pose,
        lr_bg_rot                   = args.lr_bg_rot,
        lr_bg_trans                 = args.lr_bg_trans,
        bg_max_points               = args.bg_max_points,
        background_pose_init_dir    = args.background_pose_init_dir,
        bg_init_stride              = args.bg_init_stride,
        bg_voxel_size               = args.bg_voxel_size,
        n_obj_gaussians             = args.n_obj_gaussians,
        n_hand_gaussians            = args.n_hand_gaussians,
        use_cosine_lr_schedule      = args.use_cosine_lr_schedule,
        cosine_lr_min_ratio         = args.cosine_lr_min_ratio,
        coarse_init_scale_factor    = args.coarse_init_scale_factor,
        coarse_decay_epochs         = args.coarse_decay_epochs,
        pose_confidence_decay       = args.pose_confidence_decay,
        pose_confidence_ref_frame   = args.pose_confidence_ref_frame,
        pose_confidence_dynamic_tau = args.pose_confidence_dynamic_tau,
        w_pose_init_prior           = args.w_pose_init_prior,
        rotation_search_n_candidates= args.rotation_search_n_candidates,
        rotation_search_period      = args.rotation_search_period,
        rotation_search_local_frac  = args.rotation_search_local_frac,
        rotation_search_local_max_deg = args.rotation_search_local_max_deg,
        rotation_search_silhouette_weight = args.rotation_search_silhouette_weight,
        rotation_search_smoothness_weight = args.rotation_search_smoothness_weight,
        use_l2_photometric                = args.use_l2_photometric,
        use_l2_silhouette                 = args.use_l2_silhouette,
        train_resolution_scale            = args.train_resolution_scale,
        multiview_include_background      = args.multiview_include_background,
        checkpoint_path                   = args.checkpoint_path,
        resume_from_checkpoint            = args.resume_from_checkpoint,
        ignore_optimizer_state            = args.ignore_optimizer_state,
        freeze_gaussians                  = args.freeze_gaussians,
        random_init_obj_pose              = args.random_init_obj_pose,
        random_init_obj_pose_trans_std    = args.random_init_obj_pose_trans_std,
        freeze_bg_rot               = args.freeze_bg_rot,
        freeze_bg_trans             = args.freeze_bg_trans,
        weights = LossWeights(
            photometric       = args.w_photometric,
            silhouette        = args.w_silhouette,
            silhouette_obj    = args.w_silhouette_obj,
            silhouette_hand   = args.w_silhouette_hand,
            depth             = args.w_depth,
            smooth_obj_rot     = args.w_smooth_obj_rot,
            smooth_obj_trans   = args.w_smooth_obj_trans,
            smooth_hand_rot    = args.w_smooth_hand_rot,
            smooth_hand_finger = args.w_smooth_hand_finger,
            smooth_hand_trans  = args.w_smooth_hand_trans,
            smooth_bg_rot      = args.w_smooth_bg_rot,
            smooth_bg_trans    = args.w_smooth_bg_trans,
            beta_prior        = args.w_beta_prior,
            hand_scale_prior  = args.w_hand_scale_prior,
            delta_p_reg       = args.w_delta_p_reg,
            obj_scale_prior   = args.w_obj_scale_prior,
        ),
        device                      = args.device,
        seed                        = args.seed,
    )


if __name__ == "__main__":
    main()
