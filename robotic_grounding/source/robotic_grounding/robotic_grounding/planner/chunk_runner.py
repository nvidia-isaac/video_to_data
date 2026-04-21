"""Chunked autoregressive inference and stitching.

Ported 1:1 from the original chunk_runner. Uses the bundled inferencer's
predict() method for the neural network call and the bundled converter
for features → qpos conversion.
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import torch

from robotic_grounding.planner.data_adapters import extract_feature_from_bones_rep
from robotic_grounding.planner.smoothing import hamming_smooth, smooth_qpos


def run_one_chunk(
    inferencer,
    models,
    start_global,
    end_global,
    end_local,
    gt_tokens,
    num_inference_steps=1,
    static_pose_cond=None,
    ee_only_no_root=False,
    start_root_only=False,
    ee_override=None,
    ee_override_kf_start=None,
    ee_override_kf_end=None,
):
    """Run inference for one chunk between start and end keyframes.

    Args:
        inferencer: Bundled inferencer with predict() method.
        models: dict with 'pose' and 'root' model objects.
        start_global: [1, >=nfpt, D] tail of previous prediction or GT seed (unnormalized).
        end_global: [1, nfpt, D] GT global features at end waypoint (unnormalized).
        end_local: [1, nfpt, D] GT local features at end waypoint (unnormalized).
        gt_tokens: Number of tokens this chunk should span.
        ee_override: Optional [1, T_total, 18] external EE override tensor.
        ee_override_kf_start: Frame index into ee_override for start keyframe.
        ee_override_kf_end: Frame index into ee_override for end keyframe.

    Returns:
        (pred_global, pred_tokens): predicted features [1, frames, D] and token count.
    """
    device = start_global.device
    nfpt = models["pose"].mmm_net.get_num_frames_per_token()
    global_motion_rep = inferencer.global_motion_rep
    local_motion_rep = inferencer.local_motion_rep
    motion_rep = inferencer.motion_rep

    # Convert start global → local
    start_local = motion_rep.dual_rep.global_to_local(
        start_global,
        is_normalized=False,
        to_normalize=False,
        lengths=torch.full([1], start_global.shape[1], device=device),
    )

    # Assemble 2*nfpt keyframe window: [start_tail | end_target]
    kf_global = torch.cat([start_global[:, -nfpt:], end_global], dim=1)
    kf_local = torch.cat([start_local[:, -nfpt:], end_local], dim=1)
    assert kf_global.shape[1] == 2 * nfpt

    kf_idx = torch.cat([torch.arange(nfpt), torch.arange(nfpt, 2 * nfpt)])

    # Extract root features
    global_root_values = extract_feature_from_bones_rep(
        kf_global, global_motion_rep, inferencer.EXTERNAL_ROOT_FEATURE_MODE
    )[:, kf_idx, :]
    local_root_values = extract_feature_from_bones_rep(
        kf_local, local_motion_rep, inferencer.EXTERNAL_ROOT_FEATURE_MODE
    )[:, kf_idx, :]

    # Extract pose features (EE override or FK-derived)
    if ee_override is not None and inferencer._use_global_ee:
        start_ee = ee_override[:, ee_override_kf_start : ee_override_kf_start + nfpt]
        end_ee = ee_override[:, ee_override_kf_end : ee_override_kf_end + nfpt]
        local_poses = torch.cat([start_ee, end_ee], dim=1)
    else:
        local_poses = inferencer.extract_external_pose(kf_global, kf_local, kf_idx)

    # Build availability masks
    has_global_root = torch.ones_like(global_root_values[:, :, 0], dtype=torch.bool)
    has_local_root = torch.ones_like(local_root_values[:, :, 0], dtype=torch.bool)
    has_local_poses = torch.ones_like(local_poses[:, :, 0], dtype=torch.bool)
    has_global_root[:, nfpt:-2] = False
    has_local_root[:, nfpt:-2] = False
    has_local_root[:, nfpt - 1] = False
    has_local_poses[:, nfpt:-2] = False

    if start_root_only:
        has_global_root[:, nfpt:] = False
        has_local_root[:, nfpt:] = False
        has_local_poses[:, :] = True

    if ee_only_no_root:
        has_global_root[:, :] = False
        has_local_root[:, :] = False

    MASKED_NUM_TOKENS = models["root"].mmm_net.MASKED_NUM_TOKENS
    num_tokens = torch.full([1], MASKED_NUM_TOKENS, dtype=torch.int, device=device)

    config = {
        "num_inference_step": num_inference_steps,
        "smooth_root_traj": False,
        "allow_pred_out_of_reach_num_tokens": False,
        "pose_token_sampling_use_argmax": True,
    }

    # EE-specific overrides
    if inferencer._use_global_ee:
        _pose_matches = getattr(inferencer, "_pose_ee_matches_root", False)
        _vqvae_uses_ee_target = (
            inferencer._vqvae_pose_model.decoder_target_cond_feature_mode
            == "global_ee_pose"
        )

        config["full_local_poses_at_keyframes"] = kf_local[:, kf_idx, :]
        config["full_global_poses_at_keyframes"] = kf_global[:, kf_idx, :]
        config["has_full_local_poses"] = has_local_poses

        if _vqvae_uses_ee_target and ee_override is None:
            config["use_constraints_at_decoder"] = True
        else:
            config["use_constraints_at_decoder"] = False

        if not _pose_matches:
            pose_jg = models["pose"].args.get("global_ee_joint_groups")
            kf_local_normed = local_motion_rep.normalize(kf_local)
            override_lp = extract_feature_from_bones_rep(
                kf_local_normed, local_motion_rep, "ee_pose", joint_groups=pose_jg
            )[:, kf_idx, :]
            config["override_local_poses_for_pose"] = override_lp

    pred_global, num_pred = inferencer.predict(
        global_root_values,
        has_global_root,
        local_root_values,
        has_local_root,
        local_poses,
        has_local_poses,
        num_tokens,
        config=config,
        info={},
    )

    pred_tok = num_pred.item()
    return pred_global[:, : pred_tok * nfpt], pred_tok


def _run_single_pass(
    inferencer,
    models,
    global_motions,
    local_motions,
    start_frame,
    end_frame,
    chunk_tokens,
    nfpt,
    min_tokens,
    num_inference_steps,
    mode,
    static_pose_cond,
    ee_only_no_root,
    start_root_only,
    ee_override,
    gt_canon_root,
):
    """Run one pass of chunked inference."""
    w_frames = chunk_tokens * nfpt
    device = global_motions.device
    s, e = start_frame, end_frame

    predictions = [global_motions[:, s : s + nfpt].to(device)]
    pred_ranges = [(s, s + nfpt)]
    chunk_kf_info = []
    cursor = s

    with torch.no_grad():
        while cursor + nfpt < e:
            next_wp = min(cursor + w_frames, e)
            next_wp = cursor + ((next_wp - cursor) // nfpt) * nfpt

            gt_tok = (next_wp - cursor) // nfpt
            if gt_tok < min_tokens:
                break

            if mode == "gt_anchored":
                start_global = global_motions[
                    :, max(s, cursor - nfpt) : cursor + nfpt
                ].to(device)
            else:
                prev = predictions[-1]
                n_ctx = min(2 * nfpt, prev.shape[1])
                start_global = prev[:, -n_ctx:]

            end_global = global_motions[:, next_wp - nfpt : next_wp].to(device)
            end_local = local_motions[:, next_wp - nfpt : next_wp].to(device)

            pred, pred_tok = run_one_chunk(
                inferencer,
                models,
                start_global,
                end_global,
                end_local,
                gt_tokens=gt_tok,
                num_inference_steps=num_inference_steps,
                static_pose_cond=static_pose_cond,
                ee_only_no_root=ee_only_no_root,
                start_root_only=start_root_only,
                ee_override=ee_override,
                ee_override_kf_start=cursor,
                ee_override_kf_end=next_wp - nfpt,
            )

            pred_start = cursor + nfpt
            predictions.append(pred)
            pred_ranges.append((pred_start, pred_start + pred_tok * nfpt))

            chunk_kf_info.append(
                {
                    "pred_frames": pred_tok * nfpt,
                }
            )

            cursor = next_wp

    return predictions, pred_ranges, chunk_kf_info


def run_chunked_inference(
    inferencer,
    models,
    global_motions,
    local_motions,
    start_frame,
    end_frame,
    chunk_tokens,
    num_inference_steps=1,
    mode="autoregressive",
    static_pose_cond=None,
    ee_only_no_root=False,
    start_root_only=False,
    ee_override=None,
    half_stride_blend=True,
):
    """Run the full chunked inference loop over a sequence.

    Returns:
        (predictions, chunk_kf_info): list of [1, frames, D] and list of dicts.
    """
    nfpt = models["pose"].mmm_net.get_num_frames_per_token()
    min_tokens = models["root"].args["min_tokens"]
    device = global_motions.device
    global_motion_rep = inferencer.global_motion_rep

    s, e = start_frame, end_frame

    gt_canon_root = None
    if ee_override is not None:
        gt_canon_root = global_motion_rep.compute_root_pos_and_rot(
            global_motions.to(device), return_quat=False, return_angle=False
        )[0].cpu()

    pass_args = dict(
        inferencer=inferencer,
        models=models,
        global_motions=global_motions,
        local_motions=local_motions,
        start_frame=s,
        end_frame=e,
        chunk_tokens=chunk_tokens,
        nfpt=nfpt,
        min_tokens=min_tokens,
        num_inference_steps=num_inference_steps,
        static_pose_cond=static_pose_cond,
        ee_only_no_root=ee_only_no_root,
        start_root_only=start_root_only,
        ee_override=ee_override,
        gt_canon_root=gt_canon_root,
    )

    predictions, pred_ranges, chunk_kf_info = _run_single_pass(mode=mode, **pass_args)

    if not half_stride_blend:
        return predictions, chunk_kf_info

    # Half-stride blending
    half_stride = (chunk_tokens // 2) * nfpt
    offset_start = s + half_stride
    offset_start = s + ((offset_start - s) // nfpt) * nfpt
    if offset_start + min_tokens * nfpt >= e:
        return predictions, chunk_kf_info

    preds_b, ranges_b, _ = _run_single_pass(
        mode="gt_anchored", **{**pass_args, "start_frame": offset_start}
    )

    # Blend both passes in feature space with triangular weights
    T_total = global_motions.shape[1]
    D = global_motions.shape[2]
    accum = torch.zeros(1, T_total, D, device=device)
    weight = torch.zeros(1, T_total, 1, device=device)
    pass_buffers = []

    def _accumulate(preds, ranges):
        feat_buf = torch.zeros(1, T_total, D, device=device)
        w_buf = torch.zeros(1, T_total, 1, device=device)
        for pred, (rs, _re) in zip(preds, ranges, strict=False):
            n = min(pred.shape[1], T_total - rs)
            if n <= 0:
                continue
            pred_chunk = pred[:, :n]
            w = torch.linspace(0, 1, n // 2 + 1, device=device)
            if n % 2 == 0:
                w = torch.cat([w[:-1], w.flip(0)])
            else:
                w = torch.cat([w, w.flip(0)[1:]])
            w = w[:n].view(1, n, 1).clamp(min=0.1)
            accum[:, rs : rs + n] += pred_chunk.to(device) * w
            weight[:, rs : rs + n] += w
            feat_buf[:, rs : rs + n] += pred_chunk.to(device) * w
            w_buf[:, rs : rs + n] += w
        pass_buffers.append((feat_buf, w_buf))

    _accumulate(predictions, pred_ranges)
    _accumulate(preds_b, ranges_b)

    blended = accum / weight.clamp(min=1e-8)

    # Hard selection for foot contacts
    if (
        hasattr(global_motion_rep, "indices")
        and "foot_contacts" in global_motion_rep.indices
    ):
        fc_idx = global_motion_rep.indices["foot_contacts"]
        if len(pass_buffers) == 2:
            feat_a = pass_buffers[0][0] / pass_buffers[0][1].clamp(min=1e-8)
            feat_b = pass_buffers[1][0] / pass_buffers[1][1].clamp(min=1e-8)
            has_a = pass_buffers[0][1].squeeze(-1) > 0
            has_b = pass_buffers[1][1].squeeze(-1) > 0
            fc_a = feat_a[:, :, fc_idx]
            fc_b = feat_b[:, :, fc_idx]
            strength_a = fc_a.mean(dim=-1, keepdim=True)
            strength_b = fc_b.mean(dim=-1, keepdim=True)
            both = has_a & has_b
            prefer_a = (strength_a.squeeze(-1) >= strength_b.squeeze(-1)) | ~has_b
            prefer_a = prefer_a.unsqueeze(-1).expand_as(fc_a)
            fc_hard = torch.where(prefer_a, fc_a, fc_b)
            fc_hard = torch.where(
                both.unsqueeze(-1).expand_as(fc_a), fc_hard, blended[:, :, fc_idx]
            )
            blended[:, :, fc_idx] = fc_hard

    valid_frames = (weight.squeeze(-1) > 0)[0].nonzero(as_tuple=True)[0]
    if valid_frames.numel() == 0:
        return predictions, chunk_kf_info

    blend_start = valid_frames[0].item()
    blend_end = valid_frames[-1].item() + 1

    blended_preds = [
        global_motions[:, blend_start : blend_start + nfpt].to(device),
        blended[:, blend_start + nfpt : blend_end],
    ]

    return blended_preds, chunk_kf_info


# -- Stitching --


def _slerp_quat(q0, q1, alpha):
    """Slerp between quaternion tensors (xyzw format)."""
    dot = (q0 * q1).sum(dim=1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0 - 1e-6)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    w0 = torch.sin((1 - alpha) * omega) / sin_omega
    w1 = torch.sin(alpha * omega) / sin_omega
    near_parallel = sin_omega.abs() < 1e-6
    w0 = torch.where(near_parallel, 1 - alpha, w0)
    w1 = torch.where(near_parallel, alpha, w1)
    result = w0 * q0 + w1 * q1
    return result / result.norm(dim=1, keepdim=True).clamp(min=1e-8)


def stitch_predictions(predictions, converter, motion_rep, nfpt, smooth=True):
    """Stitch prediction chunks into a single qpos tensor.

    Converts each chunk to qpos, then blends overlapping regions with
    smoothstep interpolation and Slerp for root quaternions.

    Returns:
        pred_qpos: [T, 36] tensor, or None.
    """
    pred_chunks = [p for p in predictions[1:] if p.shape[1] > 0]
    if not pred_chunks:
        return None

    qpos_chunks = []
    for p in pred_chunks:
        q = (
            converter.convert_mfm_features_to_mujoco_qpos(
                p, motion_rep, is_normalized=False
            )[0]
            .cpu()
            .detach()
        )
        qpos_chunks.append(q)

    overlap = nfpt
    stitched_parts = []
    boundary_indices = []
    running_len = 0
    for i, qchunk in enumerate(qpos_chunks):
        if i == 0:
            stitched_parts.append(qchunk)
            running_len = qchunk.shape[0]
        else:
            lin = torch.linspace(0, 1, overlap)
            alpha = (3 * lin**2 - 2 * lin**3)[:, None]
            prev_tail = stitched_parts[-1][-overlap:]
            curr_head = qchunk[:overlap]

            blended_quat = _slerp_quat(
                prev_tail[:, 3:7], curr_head[:, 3:7], alpha[:, :1]
            )
            blended = prev_tail * (1 - alpha) + curr_head * alpha
            blended[:, 3:7] = blended_quat

            boundary_idx = running_len - overlap + overlap // 2
            boundary_indices.append(boundary_idx)

            stitched_parts[-1] = stitched_parts[-1][:-overlap]
            running_len -= overlap
            stitched_parts.append(blended)
            running_len += overlap
            stitched_parts.append(qchunk[overlap:])
            running_len += qchunk.shape[0] - overlap

    pred_qpos = torch.cat(stitched_parts, dim=0)

    if smooth:
        pred_qpos = smooth_qpos(pred_qpos)

    # Targeted boundary smoothing
    if boundary_indices:
        boundary_radius = 3 * nfpt
        joints = pred_qpos[:, 7:].clone()
        joints_smoothed = hamming_smooth(joints, boundary_radius)

        T_total = pred_qpos.shape[0]
        blend_mask = torch.zeros(T_total, 1, device=pred_qpos.device)
        sigma = float(nfpt) * 1.5
        for bi in boundary_indices:
            frame_idx = torch.arange(
                T_total, device=pred_qpos.device, dtype=torch.float32
            )
            gaussian = torch.exp(-0.5 * ((frame_idx - bi) / sigma) ** 2)
            blend_mask[:, 0] = torch.maximum(blend_mask[:, 0], gaussian)

        pred_qpos[:, 7:] = joints * (1 - blend_mask) + joints_smoothed * blend_mask

    return pred_qpos
