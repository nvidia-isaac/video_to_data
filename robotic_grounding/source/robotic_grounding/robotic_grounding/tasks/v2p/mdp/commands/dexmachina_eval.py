# Paper-exact ADD AUC metric (DexMachina, arXiv 2505.24853).
#
# Reference implementation: dexmachina/eval/compute_add.py (verified identical to
# upstream MandiZhao/dexmachina commit adae5bf).
#
# Paper formula (single source of truth for both the in-training metric and the
# eval-time metric):
#
#   load_part_verts (compute_add.py:55-79):
#     For each part p in {top, bottom}:
#       verts_p = obj.sample_mesh_vertices(part=p, num_samples=500, seed=42)
#     `sample_mesh_vertices` in dexmachina/envs/object.py:305-315 calls
#     trimesh.load(mesh_path).vertices, seeds np.random with `seed=42`, and
#     uniformly samples `num_samples` indices via np.random.choice. So this is
#     NOT area-weighted surface sampling — it is uniform sampling of the mesh's
#     stored vertex set with a fixed seed.
#
#   get_all_add (compute_add.py:82-119):
#     For each frame f and each part p, transform verts_p by the demo pose and
#     by the simulated pose; per-vertex L2 distance, mean across vertices →
#     scalar dist[p, env, f]. Returns one array of shape (n_envs, n_frames) per
#     part.
#
#   compute_auc (compute_add.py:122-129):
#     accuracies[i] = mean over all entries of (dist < threshold_i)
#     AUC = np.trapz(accuracies, x=np.linspace(0, 1, len(thresholds)))
#     (i.e. the entire (n_envs * n_frames) pool for one part is flattened, no
#     per-env aggregation.)
#
#   compute_add_stats (compute_add.py:132-150):
#     For each part p: per_part_auc[p] = compute_auc(dist[p], thresholds).
#     "overall" AUC = mean of per_part_auc across parts (top, bottom).
#
#   Thresholds (compute_add.py:208): np.arange(0.01, 0.1, 0.01) → 9 thresholds
#   from 1 cm to 9 cm.
#
# Pseudocode (the single spec both call sites must implement):
#
#   verts_b = uniform sample of mesh_b's vertex array, seed=42 (per body b)
#   for each rollout step t, each env e, each body b:
#       err[b, e, t] = mean_v || R_pol(b,e,t) @ verts_b + p_pol(b,e,t)
#                              - R_dem(b,e,t) @ verts_b - p_dem(b,e,t) ||
#   per_body_auc[b] = trapz(
#       [(flatten(err[b, :, :]) < th_i).mean() for th_i in thresholds],
#       x=linspace(0, 1, len(thresholds)),
#   )
#   AUC = mean_b(per_body_auc[b])
#
# The eval loop in dexmachina/rl/eval_rl_games.py runs for `uenv.max_episode_length`
# steps. It does NOT disable resets explicitly, but in practice the eval env is
# constructed with `early_reset_threshold=0.0` (eval_rl_games.py:159) — i.e.
# no early-termination — so all envs run a single fixed-length trajectory and
# (n_envs, n_frames) is rectangular.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

# Threshold sweep matches dexmachina/eval/compute_add.py:208
# (np.arange(0.01, 0.1, 0.01) → 9 thresholds from 1 cm to 9 cm).
DEXMACHINA_THRESHOLDS_M = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)

# Paper-aligned success-rate thresholds reused across DexMachina / ManipTrans /
# SPIDER Table 1 comparisons. SPIDER's get_success_rate.py uses 0.10 m / 0.5 rad;
# ManipTrans-style coarse SRs use 0.10 / 0.15 m and 0.3 / 0.7 rad. The two extra
# fine pos thresholds (0.025 / 0.05 m) match DexMachina's AUC sweep band so the
# eval log gives a smooth SR curve next to the AUC.
PAPER_POS_SR_THRESHOLDS_M = (0.01, 0.025, 0.05, 0.10, 0.15)
PAPER_ORI_SR_THRESHOLDS_RAD = (0.1, 0.3, 0.5, 0.7, 1.0)
# SPIDER's `get_success_rate.py` uses pos<=0.10 AND quat<=0.5 as the success
# definition (after centering positions per-clip). This pair is reported as
# `eval/spider_success_rate` so the W&B page maps 1:1 onto SPIDER Table 1.
SPIDER_POS_THRESHOLD_M = 0.10
SPIDER_ORI_THRESHOLD_RAD = 0.5


def compute_paper_eval_step_errors(
    command_term: Any,
) -> dict[str, torch.Tensor]:
    """Per-step paper-aligned tracking errors used by training-time + standalone eval.

    All tensors are kept on the command term's device and are computed *without*
    leaving the caller's existing inference_mode context. Shapes:

      pos_err   : (E, B) — per-body L2 position error, meters.
      ori_err   : (E, B) — per-body quaternion error magnitude, radians.
      bbox_err  : (E, B) — mean L2 distance over the 8 AABB corners, meters.
      spider_pos_err : (E,) — SPIDER-style centered-position error averaged over
                        non-static bodies (mirrors hand_object_commands.py:2226).
      spider_ori_err : (E,) — SPIDER-style orientation error averaged over
                        non-static bodies.

    The SPIDER-* values reuse the running mean buffers that the training-time
    metric maintains (`_spider_sim_pos_sum`, `_spider_step_count`), so they only
    make sense during the same eval pass without intervening env resets (which
    is exactly how the eval-callback collects them — resets are disabled by
    raising thresholds to 1000).

    Args:
        command_term: The `dual_hands_object_tracking_command` term.

    Returns:
        dict mapping the names above to torch.Tensor objects.
    """
    import isaaclab.utils.math as _m  # noqa: PLC0415

    pos = command_term.object_position_e  # (E, B, 3)
    quat = command_term.object_orientation_e  # (E, B, 4)
    pos_d = command_term.object_body_position_command_e  # (E, B, 3)
    quat_d = command_term.object_body_wxyz_command_e  # (E, B, 4)

    pos_err = torch.norm(pos - pos_d, dim=-1)  # (E, B)
    ori_err = _m.quat_error_magnitude(quat, quat_d)  # (E, B)

    # bbox corner error: transform the precomputed local-frame corners by the
    # policy/demo poses and take per-corner L2 distance. Mean across the 8
    # corners gives a single (E, B) number per body.
    corners_local = command_term.BBOX_CORNER_VECS  # (E, B, 8, 3)
    K = corners_local.shape[2]
    pos_e = pos.unsqueeze(2).expand(-1, -1, K, -1)
    quat_e = quat.unsqueeze(2).expand(-1, -1, K, -1)
    posd_e = pos_d.unsqueeze(2).expand(-1, -1, K, -1)
    quatd_e = quat_d.unsqueeze(2).expand(-1, -1, K, -1)
    c_p, _ = _m.combine_frame_transforms(pos_e, quat_e, corners_local)
    c_d, _ = _m.combine_frame_transforms(posd_e, quatd_e, corners_local)
    bbox_err = torch.norm(c_p - c_d, dim=-1).mean(dim=-1)  # (E, B)

    # SPIDER-style centered-position error reuses the command term's running
    # running mean buffers so the result matches the spider_obj_pos_err metric
    # logged at train time. We do not mutate any state here (the buffers are
    # advanced inside the command term's own _update_metrics during the same
    # step).
    sim_pos_mean_running = (
        command_term._spider_sim_pos_sum
        / command_term._spider_step_count.clamp(min=1.0).view(-1, 1, 1)
    )
    sim_dev = pos - sim_pos_mean_running
    ref_dev = pos_d - command_term._spider_ref_pos_mean_b.unsqueeze(0)
    spider_pos_err_per_body = torch.norm(sim_dev - ref_dev, dim=-1)  # (E, B)
    non_static = (~command_term._spider_static_body_mask).float().unsqueeze(0)  # (1, B)
    n_non_static = non_static.sum(dim=-1).clamp(min=1.0)
    spider_pos_err = (spider_pos_err_per_body * non_static).sum(
        dim=-1
    ) / n_non_static  # (E,)
    spider_ori_err = (ori_err * non_static).sum(dim=-1) / n_non_static  # (E,)

    return {
        "pos_err": pos_err,
        "ori_err": ori_err,
        "bbox_err": bbox_err,
        "spider_pos_err": spider_pos_err,
        "spider_ori_err": spider_ori_err,
    }


def summarize_paper_eval_pools(
    pos_pool_per_body: list[list[float]],
    ori_pool_per_body: list[list[float]],
    bbox_pool_per_body: list[list[float]],
    spider_pos_pool: list[float],
    spider_ori_pool: list[float],
    body_names: list[str],
    device: torch.device | str = "cpu",
    pos_sr_thresholds: tuple[float, ...] = PAPER_POS_SR_THRESHOLDS_M,
    ori_sr_thresholds: tuple[float, ...] = PAPER_ORI_SR_THRESHOLDS_RAD,
    spider_pos_threshold: float = SPIDER_POS_THRESHOLD_M,
    spider_ori_threshold: float = SPIDER_ORI_THRESHOLD_RAD,
) -> dict[str, float]:
    """Flatten paper-aligned per-step pools into the eval/* metrics dict.

    Mirrors the existing dexmachina_AUC flat-pool aggregation: pool over
    (n_envs × n_frames) per body, then aggregate across bodies. All three of
    pos/ori/bbox pools must have a matching outer length == num_bodies.

    Returns a flat dict of metric_name -> float, suitable for `wandb.log(...)`.
    Metric names use the user's requested `_mean` / `_max` suffixes and the
    `eval/object_*_per_body_<body_name>` per-body breakdowns.
    """
    out: dict[str, float] = {}
    num_bodies = len(pos_pool_per_body)
    if num_bodies == 0:
        return out
    # Truncate all pools to the same length per body (matches the existing AUC
    # `n_samples_per_body = min(...)` invariant).
    n_pos = min(len(p) for p in pos_pool_per_body)
    n_ori = min(len(p) for p in ori_pool_per_body)
    n_bbox = min(len(p) for p in bbox_pool_per_body)
    n_per_body = min(n_pos, n_ori, n_bbox)
    if n_per_body == 0:
        return out

    pos_t = torch.tensor(
        [p[:n_per_body] for p in pos_pool_per_body],
        dtype=torch.float32,
        device=device,
    )  # (B, N)
    ori_t = torch.tensor(
        [o[:n_per_body] for o in ori_pool_per_body],
        dtype=torch.float32,
        device=device,
    )
    bbox_t = torch.tensor(
        [b[:n_per_body] for b in bbox_pool_per_body],
        dtype=torch.float32,
        device=device,
    )

    # Flat-pool means (over envs × frames × bodies). Match dexmachina AUC's
    # "mean across bodies" by computing per-body means first.
    pos_per_body_mean = pos_t.mean(dim=-1)  # (B,)
    ori_per_body_mean = ori_t.mean(dim=-1)
    bbox_per_body_mean = bbox_t.mean(dim=-1)
    pos_per_body_max = pos_t.max(dim=-1).values
    ori_per_body_max = ori_t.max(dim=-1).values

    out["eval/object_position_error_mean"] = float(pos_per_body_mean.mean().item())
    out["eval/object_position_error_max"] = float(pos_per_body_max.max().item())
    out["eval/object_orientation_error_mean"] = float(ori_per_body_mean.mean().item())
    out["eval/object_orientation_error_max"] = float(ori_per_body_max.max().item())
    out["eval/object_bbox_corner_error_mean"] = float(bbox_per_body_mean.mean().item())

    # Per-body breakdowns disabled — too noisy in W&B and the top-level mean is
    # sufficient. Re-enable if cross-body comparison is needed.
    # if num_bodies > 1:
    #     safe_names = [
    #         f"obj{i}" if nm.isdigit() else nm.replace(" ", "_").replace("/", "_")
    #         for i, nm in enumerate(body_names)
    #     ]
    #     for b in range(num_bodies):
    #         nm = safe_names[b] if b < len(safe_names) else f"body{b}"
    #         out[f"eval/object_position_error_per_body_{nm}"] = float(
    #             pos_per_body_mean[b].item()
    #         )
    #         out[f"eval/object_orientation_error_per_body_{nm}"] = float(
    #             ori_per_body_mean[b].item()
    #         )
    #         out[f"eval/object_bbox_corner_error_per_body_{nm}"] = float(
    #             bbox_per_body_mean[b].item()
    #         )

    # SR threshold sweep disabled — these are the AUC integrand, captured by
    # ``eval/dexmachina_AUC_mean``. Re-enable if you need per-threshold detail.
    # pos_flat = pos_t.reshape(-1)
    # ori_flat = ori_t.reshape(-1)
    # for th in pos_sr_thresholds:
    #     label = f"{th:.3f}m" if th < 0.1 else f"{th:.2f}m"
    #     out[f"eval/sr_pos@{label}"] = float((pos_flat <= th).float().mean().item())
    # for th in ori_sr_thresholds:
    #     label = f"{th:.1f}rad"
    #     out[f"eval/sr_ori@{label}"] = float((ori_flat <= th).float().mean().item())

    # SPIDER paper success rate: pos AND ori within paper thresholds. We log
    # both the joint and the marginals so the W&B page lines up with their
    # Table 1 cells directly.
    if spider_pos_pool and spider_ori_pool:
        n_spider = min(len(spider_pos_pool), len(spider_ori_pool))
        if n_spider > 0:
            sp_pos = torch.tensor(
                spider_pos_pool[:n_spider], dtype=torch.float32, device=device
            )
            sp_ori = torch.tensor(
                spider_ori_pool[:n_spider], dtype=torch.float32, device=device
            )
            out["eval/spider_obj_pos_err_mean"] = float(sp_pos.mean().item())
            out["eval/spider_obj_ori_err_mean"] = float(sp_ori.mean().item())
            pos_ok = (sp_pos <= spider_pos_threshold).float()
            ori_ok = (sp_ori <= spider_ori_threshold).float()
            out["eval/spider_sr_pos@0.10m"] = float(pos_ok.mean().item())
            out["eval/spider_sr_ori@0.5rad"] = float(ori_ok.mean().item())
            out["eval/spider_success_rate"] = float((pos_ok * ori_ok).mean().item())

    return out


# Module-level cache: (resolved_mesh_path, num_samples, seed) -> (N, 3) np.float32
# Keyed by the resolved absolute mesh path so re-sampling within a run is
# deterministic and we don't re-load the mesh on every command-term init.
_VERT_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def _mesh_unit_scale(mesh_path: Path, vertices: np.ndarray) -> float:
    """Return the scale needed to express mesh vertices in simulation meters."""
    name = mesh_path.name.lower()
    if name.endswith(("_cm.obj", "_cm.ply", "_cm.stl")):
        return 0.01

    # Manipulation objects should not span many meters in the local body frame.
    # This catches centimeter-authored assets even when the filename does not
    # carry the `_cm` suffix.
    if vertices.size:
        extent = np.ptp(vertices, axis=0)
        if float(np.nanmax(extent)) > 2.0:
            return 0.01
    return 1.0


def sample_mesh_surface_vertices(
    mesh_path: str | Path,
    num_samples: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Paper-exact mesh vertex sampler — uniform over the stored vertex set, seeded.

    Mirrors dexmachina/envs/object.py:305-315 (`sample_mesh_vertices`):

        mesh = trimesh.load(mesh_fname)
        np.random.seed(seed)
        idxs = np.random.choice(num_vertices, num_samples, replace=replace)
        return vertices[idxs]

    The paper does NOT use area-weighted surface sampling; it uniformly samples
    from the mesh's stored vertex array (`mesh.vertices`). With seed=42 this is
    deterministic across calls for the same mesh. Some local retargeted assets
    store centimeter vertices in files named ``*_cm.*`` while the Isaac body
    pose and URDF meshes are in meters; those vertices are converted to meters
    before being returned.

    Args:
        mesh_path: Path to a mesh file (.obj, .stl, .ply, etc.) loadable by trimesh.
        num_samples: Number of vertices to return.
        seed: RNG seed for np.random (paper default: 42).

    Returns:
        np.ndarray of shape (num_samples, 3), float32, in the mesh's local frame.

    Raises:
        FileNotFoundError if the mesh can't be loaded.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    cache_key = (str(mesh_path.resolve()), int(num_samples), int(seed))
    cached = _VERT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"loaded object is not a Trimesh: {type(mesh)} from {mesh_path}"
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    num_vertices = vertices.shape[0]
    replace = num_samples > num_vertices
    # Match upstream exactly: seed np.random (NOT a local Generator) so behavior
    # mirrors dexmachina/envs/object.py.
    np.random.seed(int(seed))
    idxs = np.random.choice(num_vertices, num_samples, replace=replace)
    unit_scale = _mesh_unit_scale(mesh_path, vertices)
    out = np.asarray(vertices[idxs] * unit_scale, dtype=np.float32)
    _VERT_CACHE[cache_key] = out
    return out


def compute_dexmachina_auc(
    per_body_errors: torch.Tensor,
    thresholds: tuple[float, ...] = DEXMACHINA_THRESHOLDS_M,
) -> torch.Tensor:
    """Paper-exact ADD-AUC across the threshold sweep.

    Mirrors `compute_auc` + the per-part-then-overall aggregation in
    `compute_add_stats` of dexmachina/eval/compute_add.py:122-150.

    For each body b independently:
        accuracies[i] = mean over all (env × frame) entries of
                        (per_body_errors[b] < threshold_i)
        per_body_auc[b] = trapz(accuracies, x=linspace(0, 1, len(thresholds)))
    Returns: mean over bodies of per_body_auc.

    Args:
        per_body_errors: Tensor of shape (num_bodies, N) where N is a flat pool
            of per-vertex-mean ADD values across (envs × frames). N can vary
            from one body to the next only if the caller pads / masks, but in
            practice we always pass identical-length pools per body.

            Alternative accepted shape: a 1-D tensor of length N (interpreted
            as a single-body pool); a scalar AUC is returned in that case.
        thresholds: Ordered threshold sweep in meters. Default matches the paper.

    Returns:
        Scalar tensor (0-D) — the paper's "overall" AUC, ∈ [0, 1].

    Notes:
        - `np.trapz(y, x=linspace(0, 1, n))` integrates over a fixed x-range of
          [0, 1], so the AUC is normalized regardless of how many thresholds
          the caller passes.
        - If `per_body_errors` is empty along the pool dimension we return 0
          (paper formula is undefined; this keeps the metric loggable).
    """
    if per_body_errors.numel() == 0:
        return torch.zeros(
            (), device=per_body_errors.device, dtype=per_body_errors.dtype
        )

    if per_body_errors.dim() == 1:
        per_body_errors = per_body_errors.unsqueeze(0)  # (1, N)
    elif per_body_errors.dim() != 2:
        raise ValueError(
            "per_body_errors must be 1-D (N,) or 2-D (num_bodies, N); got "
            f"shape={tuple(per_body_errors.shape)}"
        )

    device = per_body_errors.device
    dtype = per_body_errors.dtype
    th = torch.tensor(thresholds, device=device, dtype=dtype)  # (T,)
    # accuracies[b, i] = mean over flat pool of (err[b, :] < th_i)
    # Broadcast: errors → (B, 1, N); th → (1, T, 1).
    accuracies = (
        (per_body_errors.unsqueeze(1) < th.view(1, -1, 1)).to(dtype).mean(dim=-1)
    )  # (B, T)
    x = torch.linspace(0.0, 1.0, len(thresholds), device=device, dtype=dtype)
    per_body_auc = torch.trapezoid(accuracies, x=x, dim=-1)  # (B,)
    return per_body_auc.mean()
