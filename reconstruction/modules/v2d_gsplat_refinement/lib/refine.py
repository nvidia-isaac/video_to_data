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
from PIL import Image
from tqdm import tqdm

from .background import (
    BackgroundGaussians,
    BackgroundPoseField,
    init_background_from_depth,
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
    photometric_loss,
    rotation_smoothness,
    silhouette_loss,
    temporal_smoothness,
)
from .pose_fields import HandPoseField, ObjectPoseField
from .render import render_rgb_depth


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
    lr_object_pose: float = 1e-3,
    lr_hand_pose: float = 1e-3,
    lr_betas: float = 1e-4,
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
    lr_bg_pose: float = 1e-3,
    bg_max_points: int = 50000,
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

    hand_slots: list[HandSlot] = []
    for side, ht, md, out_pose in hand_tracks:
        pf = HandPoseField(ht, mano_assets_root, device=device).to(device)
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

        ref_rgb   = cache.rgb[ref_t].to(device)
        ref_obj   = cache.obj_mask[ref_t].to(device)
        ref_union = ref_obj.clone()
        for hm in cache.hand_masks:
            ref_union = torch.maximum(ref_union, hm[ref_t].to(device))
        ref_depth = cache.depth[ref_t].to(device)            # may contain +inf
        K_intr    = K

        anchors, colors, init_scales = init_background_from_depth(
            rgb         = ref_rgb,
            depth       = ref_depth,
            union_mask  = ref_union,
            K           = K_intr,
            max_points  = bg_max_points,
        )
        print(f"Background: {anchors.shape[0]} Gaussians initialized.")

        bg_gaussians  = BackgroundGaussians(anchors, colors, init_scales).to(device)
        bg_pose_field = BackgroundPoseField(
            n_frames=len(frame_indices), device=device, ref_frame_t=ref_t
        ).to(device)

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

    param_groups: list[dict] = []
    param_groups += _gaussian_groups(obj_gaussians, lr_gaussians)
    param_groups.append({
        "params": [obj_pose_field.axis_angle, obj_pose_field.translation],
        "lr": lr_object_pose,
    })
    for slot in hand_slots:
        # Hand Gaussians get their own base LR knob — they typically need a bump
        # over the object's because the hand covers fewer pixels in image and
        # gradient signal per-Gaussian is correspondingly diluted.
        param_groups += _gaussian_groups(slot.gaussians, lr_hand_gaussians)
        param_groups.append({
            "params": [slot.pose_field.global_orient,
                       slot.pose_field.hand_pose,
                       slot.pose_field.cam_t],
            "lr": lr_hand_pose,
        })
        param_groups.append({"params": [slot.pose_field.betas], "lr": lr_betas})
    if bg_gaussians is not None:
        param_groups += _gaussian_groups(bg_gaussians, lr_bg_gaussians)
        param_groups.append({
            "params": [bg_pose_field.axis_angle, bg_pose_field.translation],
            "lr": lr_bg_pose,
        })
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

    # Dump step 0: the initialization (gray object color, skin-tone hands,
    # poses straight from FoundationPose / HaMeR-aligned). Lets you visually
    # confirm the init before the optimizer starts moving things.
    if render_every > 0:
        _dump_progress_frame(
            progress_dir, 0, debug_frame_t, cache,
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

            # Per-frame losses, summed across the batch then backward once.
            for t in batch_ts:
                t = int(t)
                sup = cache.get(t, device)
                obj_mask = sup["obj_mask"]
                hand_msks = sup["hand_masks"]
                union_mask = obj_mask.clone()
                for m in hand_msks:
                    union_mask = torch.maximum(union_mask, m)

                R_obj, t_obj = obj_pose_field(t)
                obj_frame    = obj_gaussians(R_obj, t_obj)
                hand_frames  = []
                for slot in hand_slots:
                    verts_cam, R_per_vert = slot.pose_field.posed_verts_and_rotmats_camera(t)
                    hand_frames.append(slot.gaussians(verts_cam, R_per_vert))

                # Optional background: rigid Gaussian set in world frame
                # transformed by per-frame world→camera pose.
                bg_frame = None
                if bg_gaussians is not None:
                    R_bg, t_bg = bg_pose_field(t)
                    bg_frame = bg_gaussians(R_bg, t_bg)

                all_frames = [obj_frame] + hand_frames
                if bg_frame is not None:
                    all_frames.append(bg_frame)
                combined = concat_frames(all_frames)
                if coarse_scale_mul != 1.0:
                    combined = dataclasses.replace(
                        combined, scales=combined.scales * coarse_scale_mul
                    )

                # Per-Gaussian class one-hot labels: object → class 0,
                # hand i → class i+1. Background Gaussians get an all-zero
                # label vector so they contribute RGB but not class
                # probability — at a background pixel the target one-hot
                # is also zeros, so L1 is satisfied with zero gradient on
                # the background's class channels.
                K_classes = 1 + len(hand_slots)
                label_chunks = [obj_frame.means.new_zeros(
                    obj_frame.means.shape[0], K_classes)]
                label_chunks[0][:, 0] = 1.0
                for i, hf in enumerate(hand_frames):
                    chunk = hf.means.new_zeros(hf.means.shape[0], K_classes)
                    chunk[:, i + 1] = 1.0
                    label_chunks.append(chunk)
                if bg_frame is not None:
                    label_chunks.append(
                        bg_frame.means.new_zeros(bg_frame.means.shape[0], K_classes)
                    )
                labels = torch.cat(label_chunks, dim=0)            # (sum_N, K)

                rgb_pred, depth_pred, _, class_pred = render_rgb_depth(
                    combined, K, W, H, extra_features=labels)

                # Photometric:
                #   - balance_photometric_by_mask: per-entity inverse-area
                #     weighting so each entity (object, each hand, optional
                #     background) contributes equally regardless of pixel
                #     count.
                #   - else if bg_gaussians: full-image L1 (background
                #     Gaussians explain non-foreground; bg dominates by
                #     pixel count).
                #   - else: foreground-only L1 (or black-bg masked variant).
                if balance_photometric_by_mask:
                    fl = weights.photometric * balanced_photometric_loss(
                        rgb_pred, sup["rgb"],
                        obj_mask, hand_msks,
                        include_background=(bg_gaussians is not None),
                    )
                elif bg_gaussians is not None:
                    fl = weights.photometric * (rgb_pred - sup["rgb"]).abs().mean()
                else:
                    fl = weights.photometric * photometric_loss(
                        rgb_pred, sup["rgb"], union_mask,
                        mask_background_to_black=mask_background_to_black,
                    )

                # Class-label silhouette: target = (H, W, K) one-hot from
                # SAM2 masks. Background pixels are all-zero (no class).
                target_class = obj_mask.new_zeros((H, W, K_classes))
                target_class[..., 0] = obj_mask
                for i, hmask in enumerate(hand_msks):
                    target_class[..., i + 1] = hmask
                fl = fl + weights.silhouette * silhouette_loss(
                    class_pred, target_class,
                    class_weights=silhouette_class_weights,
                )

                if sup["depth"] is not None:
                    fl = fl + weights.depth * depth_loss(
                        depth_pred, sup["depth"], union_mask)

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
    # Bake the learned global object scale into the per-frame Transform3d
    # scale field so downstream renderers (which apply Transform3d to the
    # original mesh) see the correct size without needing to know about
    # _log_scale_global.
    s_obj_learned = float(obj_gaussians.object_scale().detach())
    print(f"Learned object scale: {s_obj_learned:.4f}")
    refined_obj_track = obj_pose_field.export_track()
    refined_obj_track.scales = refined_obj_track.scales * s_obj_learned
    save_object_poses(refined_obj_track, refined_object_poses_dir)
    print(f"Wrote refined object poses → {refined_object_poses_dir}")
    for slot in hand_slots:
        if slot.output_pose_dir is None:
            continue
        save_hand_poses(slot.pose_field.export_track(slot.track.raw_records),
                        slot.output_pose_dir)
        print(f"Wrote refined {slot.side}-hand poses → {slot.output_pose_dir}")

    # Final overlay video — streamed render + ffmpeg encode in one pass.
    _render_overlay_video_streaming(
        cache, frame_indices, K, W, H, device,
        obj_gaussians, obj_pose_field, hand_slots,
        output_path=overlay_path,
        bg_gaussians=bg_gaussians, bg_pose_field=bg_pose_field,
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


def _render_overlay_video_streaming(
    cache, frame_indices, K, W, H, device,
    obj_gaussians, obj_pose_field, hand_slots,
    output_path: str,
    bg_gaussians=None, bg_pose_field=None,
    fps: float = 30.0,
) -> None:
    """Render + encode the overlay video in a single streaming pass.

    Avoids two big costs of the old two-stage path:
      - Re-reading source RGB from disk each frame (use the in-memory cache).
      - Writing each frame to a temp PNG and then re-decoding it for ffmpeg
        (pipe raw rgb24 bytes to ffmpeg's stdin instead).

    For a 500-frame 720p clip this typically drops overlay export from
    1–3 minutes to under 10 seconds.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}",
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
            for t, _fidx in enumerate(frame_indices):
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
                combined = concat_frames(all_frames)
                rgb_pred, _, alpha, _ = render_rgb_depth(combined, K, W, H)
                if bg_gaussians is not None:
                    mix = rgb_pred
                else:
                    mix = rgb * (1.0 - alpha.unsqueeze(-1)) + rgb_pred * alpha.unsqueeze(-1)
                arr = (mix.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                arr = np.ascontiguousarray(arr)
                proc.stdin.write(arr.tobytes())
                pbar.update(1)
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()
        pbar.close()
    print(f"Wrote overlay video → {output_path}")


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
    p.add_argument("--lr_object_pose",  type=float, default=1e-3)
    p.add_argument("--lr_hand_pose",    type=float, default=1e-3)
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
    p.add_argument("--lr_bg_pose",          type=float, default=1e-3)
    p.add_argument("--bg_max_points",       type=int,   default=50000,
                   help="Subsample background point cloud to this size.")
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
        lr_hand_pose                = args.lr_hand_pose,
        lr_betas                    = args.lr_betas,
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
        bg_max_points               = args.bg_max_points,
        n_obj_gaussians             = args.n_obj_gaussians,
        n_hand_gaussians            = args.n_hand_gaussians,
        use_cosine_lr_schedule      = args.use_cosine_lr_schedule,
        cosine_lr_min_ratio         = args.cosine_lr_min_ratio,
        coarse_init_scale_factor    = args.coarse_init_scale_factor,
        coarse_decay_epochs         = args.coarse_decay_epochs,
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
            delta_p_reg       = args.w_delta_p_reg,
            obj_scale_prior   = args.w_obj_scale_prior,
        ),
        device                      = args.device,
        seed                        = args.seed,
    )


if __name__ == "__main__":
    main()
