"""
Entity-aware Gaussian density control.

Clone/split Gaussians with high position gradient norms.
Prune Gaussians with low opacity.
New Gaussians always inherit entity_id (and skinning attributes) from their parent.

All entities (background, body, object) are eligible for densification.
Body and object Gaussians are protected from opacity-based pruning since they
may be legitimately occluded (low opacity) but still needed for the canonical
representation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from v2d.gsplat.lib.scene import GaussianScene, ENTITY_BODY, ENTITY_BACKGROUND, ENTITY_OBJECT_BASE


def _clone_gaussians(scene: GaussianScene, clone_mask: torch.Tensor) -> GaussianScene:
    """
    Clone Gaussians indicated by clone_mask: duplicate with small position jitter.
    Returns a new GaussianScene containing the cloned Gaussians only.
    """
    with torch.no_grad():
        pos = scene._positions[clone_mask].clone()
        # Jitter by a fraction of scale
        scales = torch.exp(scene._log_scales[clone_mask])
        jitter = (torch.randn_like(pos) * scales * 0.3)
        pos_new = pos + jitter

        colors = torch.clamp(
            scene._sh_dc[clone_mask, 0, :] * 0.28209479177387814 + 0.5,
            0, 1,
        )
        eids = scene.entity_ids[clone_mask]

        # Body skinning weights for cloned Gaussians
        body_clone_mask = (eids == ENTITY_BODY)
        sw_new = None
        vid_new = None
        if scene._skinning_weights_raw is not None and body_clone_mask.any():
            # Map from clone_mask indices to body-array indices
            body_mask = scene.body_mask()
            body_indices_in_clone = clone_mask[body_mask]  # body Gaussians selected for cloning
            if body_indices_in_clone.any():
                sw_new = scene._skinning_weights_raw[
                    _body_local_indices(clone_mask, body_mask)
                ].clone()
                vid_new = scene.smpl_vertex_ids[
                    _body_local_indices(clone_mask, body_mask)
                ].clone()

        cloned = GaussianScene(pos_new, colors, eids, sw_new, vid_new)
        # Copy all raw params
        cloned._rotations.data = scene._rotations[clone_mask].clone()
        cloned._log_scales.data = scene._log_scales[clone_mask].clone()
        cloned._opacities_raw.data = scene._opacities_raw[clone_mask].clone()
        cloned._sh_dc.data = scene._sh_dc[clone_mask].clone()
        cloned._sh_rest.data = scene._sh_rest[clone_mask].clone()

    return cloned


def _split_gaussians(scene: GaussianScene, split_mask: torch.Tensor, n_splits: int = 2) -> GaussianScene:
    """
    Split large Gaussians into n_splits smaller ones distributed along their major axis.
    Returns a new GaussianScene with (n_splits * n_split_gaussians) Gaussians.
    """
    with torch.no_grad():
        pos = scene._positions[split_mask]
        scales = torch.exp(scene._log_scales[split_mask])  # (M, 3)
        quats = scene._rotations[split_mask]
        quats = torch.nn.functional.normalize(quats, dim=-1)

        # Sample positions from Gaussian distribution scaled by current scale
        pos_list = []
        for _ in range(n_splits):
            jitter = torch.randn_like(pos) * scales * 0.4
            pos_list.append(pos + jitter)

        pos_new = torch.cat(pos_list, dim=0)
        eids = scene.entity_ids[split_mask].repeat(n_splits)
        colors = torch.clamp(
            scene._sh_dc[split_mask, 0, :].repeat(n_splits, 1) * 0.28209479177387814 + 0.5,
            0, 1,
        )

        body_mask = scene.body_mask()
        sw_new = None
        vid_new = None
        if scene._skinning_weights_raw is not None:
            local_idx = _body_local_indices(split_mask, body_mask)
            if local_idx is not None and len(local_idx) > 0:
                sw_new = scene._skinning_weights_raw[local_idx].repeat(n_splits, 1)
                vid_new = scene.smpl_vertex_ids[local_idx].repeat(n_splits)

        split_scene = GaussianScene(pos_new, colors, eids, sw_new, vid_new)
        # Repeat raw params n_splits times and reduce scale
        scale_factor = 1.0 / (n_splits ** 0.5)
        split_scene._rotations.data = scene._rotations[split_mask].repeat(n_splits, 1)
        split_scene._log_scales.data = (
            scene._log_scales[split_mask] + np.log(scale_factor)
        ).repeat(n_splits, 1)
        split_scene._opacities_raw.data = scene._opacities_raw[split_mask].repeat(n_splits, 1)
        split_scene._sh_dc.data = scene._sh_dc[split_mask].repeat(n_splits, 1, 1)
        split_scene._sh_rest.data = scene._sh_rest[split_mask].repeat(n_splits, 1, 1)

    return split_scene


def _body_local_indices(global_mask: torch.Tensor, body_mask: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Given a global boolean mask and the global body_mask, return the local body indices
    of Gaussians that are both in global_mask and in body_mask.
    Returns None if no overlap.
    """
    overlap = global_mask & body_mask
    if not overlap.any():
        return None
    # Compute local index within body Gaussians
    body_global_indices = body_mask.nonzero(as_tuple=False).squeeze(-1)
    overlap_global_indices = overlap.nonzero(as_tuple=False).squeeze(-1)
    # Map overlap global indices to local body indices
    local = torch.searchsorted(body_global_indices, overlap_global_indices)
    return local


def densify_and_prune(
    scene: GaussianScene,
    pos_grad_accum: torch.Tensor,  # (N,) accumulated position gradient norms
    grad_threshold: float = 0.0002,
    prune_opacity_threshold: float = 0.005,
    max_scene_extent: float = 10.0,
    max_gaussians: int = 500_000,
) -> GaussianScene:
    """
    Entity-aware densification + pruning pass.

    - All Gaussians (background, body, object) are densification candidates.
    - Body and object Gaussians are protected from opacity-based pruning.
    Returns a new GaussianScene with the updated set of Gaussians.
    """
    device = scene._positions.device

    # All entities are eligible for cloning/splitting based on gradient magnitude.
    densify_candidates = pos_grad_accum > grad_threshold

    # Split large Gaussians; clone small ones
    scales = torch.exp(scene._log_scales).max(dim=-1).values  # (N,) max scale per Gaussian
    split_threshold = max_scene_extent * 0.01

    split_mask = densify_candidates & (scales > split_threshold)
    clone_mask = densify_candidates & (scales <= split_threshold)

    parts = [scene]
    if split_mask.any():
        parts.append(_split_gaussians(scene, split_mask))
    if clone_mask.any():
        parts.append(_clone_gaussians(scene, clone_mask))

    new_scene = GaussianScene.concat(parts) if len(parts) > 1 else scene

    # Prune low-opacity background Gaussians only.
    # Body Gaussians are SMPL-grounded — occluded parts get their opacity pushed
    # down by the 2D entity mask loss (projects behind body = outside mask = BCE
    # drives alpha→0), and would be wrongly removed. Trust SMPL to place them
    # correctly; they will recover opacity once they become visible.
    # Object Gaussians are mesh-grounded for the same reason.
    with torch.no_grad():
        opacities = new_scene.opacities
        prune_eligible = ~new_scene.body_mask()
        for rid in range(new_scene.n_objects()):
            prune_eligible = prune_eligible & ~new_scene.object_mask(rid)
        keep = ~prune_eligible | (opacities > prune_opacity_threshold)

    if not keep.all():
        new_scene = _filter_scene(new_scene, keep)

    # Hard cap on total Gaussians
    if new_scene.num_gaussians > max_gaussians:
        with torch.no_grad():
            opacities = new_scene.opacities
            _, top_idx = torch.topk(opacities, max_gaussians)
            keep_top = torch.zeros(new_scene.num_gaussians, dtype=torch.bool, device=device)
            keep_top[top_idx] = True
        new_scene = _filter_scene(new_scene, keep_top)

    return new_scene


def _filter_scene(scene: GaussianScene, keep: torch.Tensor) -> GaussianScene:
    """Retain only Gaussians where keep=True. Returns a new GaussianScene."""
    with torch.no_grad():
        colors = torch.clamp(scene._sh_dc[keep, 0, :] * 0.28209479177387814 + 0.5, 0, 1)
        eids = scene.entity_ids[keep]

        has_body = (eids == ENTITY_BODY).any()
        sw_new = None
        vid_new = None

        if scene._skinning_weights_raw is not None and has_body:
            body_mask_orig = scene.body_mask()
            keep_body_global = keep & body_mask_orig
            if keep_body_global.any():
                local_idx = _body_local_indices(keep_body_global, body_mask_orig)
                if local_idx is not None:
                    sw_new = scene._skinning_weights_raw[local_idx].clone()
                    vid_new = scene.smpl_vertex_ids[local_idx].clone()

        new_scene = GaussianScene(
            scene._positions[keep].detach().clone(),
            colors,
            eids,
            sw_new,
            vid_new,
        )
        new_scene._rotations.data = scene._rotations[keep].clone()
        new_scene._log_scales.data = scene._log_scales[keep].clone()
        new_scene._opacities_raw.data = scene._opacities_raw[keep].clone()
        new_scene._sh_dc.data = scene._sh_dc[keep].clone()
        new_scene._sh_rest.data = scene._sh_rest[keep].clone()

        # Carry over initial object positions (object Gaussians are never pruned,
        # so the stored tensors remain valid for the anchor loss)
        if scene._initial_obj_positions:
            new_scene._initial_obj_positions.update(scene._initial_obj_positions)

    return new_scene
