"""ManipTrans-paper-aligned metrics for DualHandsObjectTrackingCommand.

Separated from hand_object_commands.py so all ManipTrans-specific logic lives here:
body-ID match tables, per-body weight tensors, E_j / E_ft / E_r / E_t / E_t_root.

Reference: ManipTrans arXiv 2503.21860, Section 4.1.
"""

from __future__ import annotations

from typing import Any

import torch

# ── Per-body weight / lambda tables (module-level constants) ────────────────── #

# (weight, lambda_sq) per body-name suffix for the squared-distance r_finger kernel.
_TYPE_WEIGHTS: dict[str, tuple[float, float]] = {
    "_DP": (3.0, 200.0),
    "_MP": (2.0, 100.0),
    "_PP": (1.5, 50.0),
    "_MCP_VL": (1.2, 30.0),
    "_CMC_VL": (1.0, 20.0),
    "_MC": (1.0, 20.0),
    "_C_MC": (1.0, 20.0),
    "_elastomer": (0.5, 30.0),
    "_fingertip": (3.0, 200.0),
}

# Per-finger weight multiplier applied on top of the suffix table.
_FINGER_MULT: dict[str, float] = {
    "thumb": 1.5,
    "index": 1.3,
    "middle": 1.3,
    "ring": 1.0,
    "pinky": 1.0,
}

# Per-suffix lambda for the linear-distance kernel (active reward in rewards.py).
_TYPE_LAMBDAS: dict[str, float] = {
    "_DP": 100.0,
    "_fingertip": 100.0,
    "_MP": 40.0,
    "_PP": 50.0,
    "_MCP_VL": 30.0,
    "_CMC_VL": 30.0,
    "_MC": 20.0,
    "_C_MC": 40.0,
    "_elastomer": 30.0,
}

# Per-finger scale applied to DP/fingertip lambdas (thumb = 1.0 baseline).
_DP_FINGER_FACTORS: dict[str, float] = {
    "thumb": 1.0,
    "index": 0.9,
    "middle": 0.8,
    "ring": 0.6,
    "pinky": 0.6,
}


def init_maniptrans_body_tables(
    cmd: Any,
    side: str,
    retargeted_hand_frame_names: list[str],
) -> None:
    """Build body-ID match tables, weight tensors, and termination index tensors for one hand.

    Sets the following attributes on *cmd*:

    - ``paper_{side}_robot_body_ids``      : ``(F,)`` long — robot body indices in intersection
    - ``paper_{side}_ref_frame_indices``   : ``(F,)`` long — ref-frame indices in intersection
    - ``paper_{side}_body_weights``        : ``(F,)`` float — squared-kernel weights
    - ``paper_{side}_body_decay_rates_sq`` : ``(F,)`` float — squared-kernel lambdas (deprecated)
    - ``paper_{side}_body_lambdas_lin``    : ``(F,)`` float — linear-kernel lambdas (active)
    - ``paper_{side}_{finger}_tip_idx``    : ``(0,)`` or ``(1,)`` long per finger
    - ``paper_{side}_level1_idxs``         : ``(N_PP,)`` long — proximal-phalanx indices
    - ``paper_{side}_level2_idxs``         : ``(N_MP,)`` long — middle-phalanx indices

    Called from ``DualHandsObjectTrackingCommand._init_hand_data()`` inside the
    per-side loop, after ``retargeted_{side}_hand_frames`` is set.

    Args:
        cmd: The command term instance.
        side: ``"right"`` or ``"left"``.
        retargeted_hand_frame_names: Frame name list from retarget data for this side.
    """
    side_robot = getattr(cmd, f"{side}_robot")
    robot_body_names = list(side_robot.data.body_names)

    paper_robot_body_ids: list[int] = []
    paper_ref_frame_indices: list[int] = []
    selected_body_names: list[str] = []

    for ref_idx, frame_name in enumerate(retargeted_hand_frame_names):
        if frame_name in robot_body_names:
            paper_robot_body_ids.append(robot_body_names.index(frame_name))
            paper_ref_frame_indices.append(ref_idx)
            selected_body_names.append(frame_name)

    setattr(
        cmd,
        f"paper_{side}_robot_body_ids",
        torch.tensor(paper_robot_body_ids, dtype=torch.long, device=cmd.device),
    )
    setattr(
        cmd,
        f"paper_{side}_ref_frame_indices",
        torch.tensor(paper_ref_frame_indices, dtype=torch.long, device=cmd.device),
    )

    # ── Per-body weight / lambda tables ───────────────────────────────────── #
    body_weights_list: list[float] = []
    body_decay_list: list[float] = []
    body_lambda_lin_list: list[float] = []

    for body_name in selected_body_names:
        weight, decay = 1.0, 10.0
        for suffix, (w, d) in _TYPE_WEIGHTS.items():
            if body_name.endswith(suffix):
                weight, decay = w, d
                break
        for finger, mult in _FINGER_MULT.items():
            if finger in body_name:
                weight *= mult
                break
        body_weights_list.append(weight)
        body_decay_list.append(decay)

        lam, matched_suffix = 20.0, None
        for suffix, lam_val in _TYPE_LAMBDAS.items():
            if body_name.endswith(suffix):
                lam = lam_val
                matched_suffix = suffix
                break
        if matched_suffix in ("_DP", "_fingertip"):
            for finger, factor in _DP_FINGER_FACTORS.items():
                if finger in body_name:
                    lam *= factor
                    break
        body_lambda_lin_list.append(lam)

    setattr(
        cmd,
        f"paper_{side}_body_weights",
        torch.tensor(body_weights_list, dtype=torch.float32, device=cmd.device),
    )
    # Deprecated name kept for any downstream code that reads _decay_rates_sq.
    setattr(
        cmd,
        f"paper_{side}_body_decay_rates_sq",
        torch.tensor(body_decay_list, dtype=torch.float32, device=cmd.device),
    )
    setattr(
        cmd,
        f"paper_{side}_body_lambdas_lin",
        torch.tensor(body_lambda_lin_list, dtype=torch.float32, device=cmd.device),
    )

    # ── Per-finger termination index tables ───────────────────────────────── #
    tip_indices: dict[str, int] = {}
    level1_indices: list[int] = []
    level2_indices: list[int] = []

    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        tip_idx = -1
        for idx, bname in enumerate(selected_body_names):
            if finger in bname and bname.endswith("_fingertip"):
                tip_idx = idx
                break
        if tip_idx < 0:
            for idx, bname in enumerate(selected_body_names):
                if finger in bname and bname.endswith("_DP"):
                    tip_idx = idx
                    break
        tip_indices[finger] = tip_idx

    for idx, bname in enumerate(selected_body_names):
        if bname.endswith("_PP"):
            level1_indices.append(idx)
        elif bname.endswith("_MP"):
            level2_indices.append(idx)

    for finger, idx in tip_indices.items():
        setattr(
            cmd,
            f"paper_{side}_{finger}_tip_idx",
            (
                torch.tensor([idx], dtype=torch.long, device=cmd.device)
                if idx >= 0
                else torch.tensor([], dtype=torch.long, device=cmd.device)
            ),
        )
    setattr(
        cmd,
        f"paper_{side}_level1_idxs",
        torch.tensor(level1_indices, dtype=torch.long, device=cmd.device),
    )
    setattr(
        cmd,
        f"paper_{side}_level2_idxs",
        torch.tensor(level2_indices, dtype=torch.long, device=cmd.device),
    )

    # One-time diagnostic (right hand only to avoid duplicated output).
    if side == "right":
        print(
            f"[maniptrans_metrics] paper r_finger weights for {side} hand "
            f"({len(selected_body_names)} bodies):"
        )
        for bname, w, d, lam in zip(  # noqa: B905
            selected_body_names,
            body_weights_list,
            body_decay_list,
            body_lambda_lin_list,
        ):
            print(
                f"  {bname:40s}  w={w:5.2f}  lambda_sq={d:7.2f}  lambda_lin={lam:7.2f}"
            )
        print(
            f"[maniptrans_metrics] {side} hand finger classification "
            f"(indices into selected_body_names):"
        )
        for finger, idx in tip_indices.items():
            bname = selected_body_names[idx] if idx >= 0 else "<missing>"
            print(f"  {finger}_tip -> idx={idx:3d}  ({bname})")
        print(
            f"  level_1 (PP, {len(level1_indices)} bodies): "
            f"{[selected_body_names[i] for i in level1_indices]}"
        )
        print(
            f"  level_2 (MP, {len(level2_indices)} bodies): "
            f"{[selected_body_names[i] for i in level2_indices]}"
        )


def register_maniptrans_metric_keys(cmd: Any) -> None:
    """Register ManipTrans paper metric keys in ``cmd.metrics``.

    Called once from ``DualHandsObjectTrackingCommand._init_metrics()``.
    """
    z = torch.zeros(cmd.num_envs, device=cmd.device)
    for side in ("right", "left"):
        cmd.metrics[f"{side}_hand_joint_position_error_cm"] = z.clone()
        cmd.metrics[f"{side}_hand_fingertip_position_error_cm"] = z.clone()
    for key in (
        "paper_Er_deg",
        "paper_Et_cm",
        "paper_Et_root_cm",
        "paper_Ej_cm",
        "paper_Eft_cm",
        "maniptrans_Ej_pass_left",
        "maniptrans_Ej_pass_right",
        "maniptrans_Ej_pass_both",
        "maniptrans_Eft_pass_left",
        "maniptrans_Eft_pass_right",
        "maniptrans_Eft_pass_both",
        "maniptrans_Ej_and_Eft_pass_left",
        "maniptrans_Ej_and_Eft_pass_right",
        "maniptrans_Ej_and_Eft_pass_both",
    ):
        cmd.metrics[key] = z.clone()


def update_maniptrans_metrics(
    cmd: Any,
    shared: dict[str, torch.Tensor],
) -> None:
    """Compute ManipTrans E_j, E_ft, E_r, E_t, E_t_root and write to ``cmd.metrics``.

    Called once per step from ``DualHandsObjectTrackingCommand._update_metrics()``,
    after :func:`spider_metrics.update_spider_metrics` has run and returned *shared*.

    Args:
        cmd: The command term instance.
        shared: Intermediates from :func:`spider_metrics.update_spider_metrics`:
            ``cur_sim_pos``, ``cur_ref_pos``, ``rot_err_per_body``,
            ``non_static``, ``n_non_static``, ``spider_rot_err``.
    """
    cur_sim_pos = shared["cur_sim_pos"]
    cur_ref_pos = shared["cur_ref_pos"]
    non_static = shared["non_static"]
    n_non_static = shared["n_non_static"]
    spider_rot_err = shared["spider_rot_err"]

    right_ref_frames_e = cmd.retargeted_right_hand_frames[cmd.timestep_counter]
    left_ref_frames_e = cmd.retargeted_left_hand_frames[cmd.timestep_counter]
    env_origins_b = cmd._env.scene.env_origins.unsqueeze(1)  # (E, 1, 3)
    right_robot_bodies_e = cmd.right_robot.data.body_link_pos_w - env_origins_b
    left_robot_bodies_e = cmd.left_robot.data.body_link_pos_w - env_origins_b

    # E_j: mean per-joint L2 error (cm), bimanual mean (bih = 0.5*(rh+lh)).
    right_robot_joints = right_robot_bodies_e[:, cmd.paper_right_robot_body_ids]
    right_ref_joints = right_ref_frames_e[:, cmd.paper_right_ref_frame_indices, :3]
    right_Ej_cm = (
        torch.linalg.norm(right_robot_joints - right_ref_joints, dim=-1).mean(dim=-1)
        * 100.0
    )
    left_robot_joints = left_robot_bodies_e[:, cmd.paper_left_robot_body_ids]
    left_ref_joints = left_ref_frames_e[:, cmd.paper_left_ref_frame_indices, :3]
    left_Ej_cm = (
        torch.linalg.norm(left_robot_joints - left_ref_joints, dim=-1).mean(dim=-1)
        * 100.0
    )
    cmd.metrics["right_hand_joint_position_error_cm"] = right_Ej_cm
    cmd.metrics["left_hand_joint_position_error_cm"] = left_Ej_cm

    # E_ft: mean per-fingertip L2 error (cm), M=5 per hand (*_DP bodies).
    right_ref_tips = right_ref_frames_e[:, cmd.retargeted_right_fingertip_indices, :3]
    right_Eft_cm = (
        torch.linalg.norm(
            cmd.right_hand_fingertip_position_e - right_ref_tips, dim=-1
        ).mean(dim=-1)
        * 100.0
    )
    left_ref_tips = left_ref_frames_e[:, cmd.retargeted_left_fingertip_indices, :3]
    left_Eft_cm = (
        torch.linalg.norm(
            cmd.left_hand_fingertip_position_e - left_ref_tips, dim=-1
        ).mean(dim=-1)
        * 100.0
    )
    cmd.metrics["right_hand_fingertip_position_error_cm"] = right_Eft_cm
    cmd.metrics["left_hand_fingertip_position_error_cm"] = left_Eft_cm

    # Paper scalar metrics in paper units (Table 1).
    raw_pos_err_per_body = torch.norm(cur_sim_pos - cur_ref_pos, dim=-1)  # (E, B)
    raw_pos_err_per_body_cm = raw_pos_err_per_body * 100.0

    # E_r: mean-over-non-static-bodies rotation error (degrees).
    paper_Er_deg = torch.rad2deg(spider_rot_err)
    # E_t: raw (no mean-subtraction) translation error (cm). Not SPIDER-style.
    paper_Et_cm = (raw_pos_err_per_body_cm * non_static).sum(dim=-1) / n_non_static

    # E_t_root: single-body root L2 error (cm). The ManipTrans paper E_t is
    # single-body (the root / object link). Body picker:
    #   • Multi-object (taco / hot3d): pick body 0 of each object group; if
    #     that body is static (e.g. fixed base), fall back to first non-static
    #     body within the same group. Average the per-object root errors so
    #     a single scalar represents the multi-object scene.
    #   • Single-object: use first non-static body (preserves the existing
    #     "first body that moves" convention, which is useful for articulated
    #     objects with a fixed base + moving lid where body 0 would otherwise
    #     trivially read 0).
    # The degenerate all-static fallback emits a one-time warning so the
    # condition is visible in logs.
    _ns_mask_b = ~cmd._spider_static_body_mask  # (B,) bool

    def _pick_root_in_range(start: int, end: int) -> int | None:
        """Body 0 of the range if non-static, else first non-static in range."""
        if start >= end:
            return None
        if bool(_ns_mask_b[start]):
            return start
        for _b in range(start + 1, end):
            if bool(_ns_mask_b[_b]):
                return _b
        return None

    B_total = raw_pos_err_per_body_cm.shape[1]
    if getattr(cmd, "_has_multi_object", False) and hasattr(
        cmd,
        "_obj1_root_body_idx",
    ):
        _obj1_start = int(cmd._obj1_root_body_idx)
        _root0 = _pick_root_in_range(0, _obj1_start)
        _root1 = _pick_root_in_range(_obj1_start, B_total)
        _roots = [r for r in (_root0, _root1) if r is not None]
        if _roots:
            paper_Et_root_cm = sum(
                raw_pos_err_per_body_cm[:, _r] for _r in _roots
            ) / len(_roots)
        else:
            if not getattr(cmd, "_warned_et_root_degenerate", False):
                print(
                    "[maniptrans] WARNING: no non-static body found across "
                    "multi-object scene; paper_Et_root_cm falling back to "
                    "multi-body mean.",
                )
                cmd._warned_et_root_degenerate = True
            paper_Et_root_cm = paper_Et_cm
    else:
        _ns_idx = torch.nonzero(_ns_mask_b, as_tuple=False).flatten()
        if _ns_idx.numel() > 0:
            _root_idx = int(_ns_idx[0].item())
            paper_Et_root_cm = raw_pos_err_per_body_cm[:, _root_idx]
        else:
            if not getattr(cmd, "_warned_et_root_degenerate", False):
                print(
                    "[maniptrans] WARNING: no non-static body found; "
                    "paper_Et_root_cm falling back to multi-body mean.",
                )
                cmd._warned_et_root_degenerate = True
            paper_Et_root_cm = paper_Et_cm

    paper_Ej_cm = 0.5 * (right_Ej_cm + left_Ej_cm)
    paper_Eft_cm = 0.5 * (right_Eft_cm + left_Eft_cm)

    cmd.metrics["paper_Er_deg"] = paper_Er_deg
    cmd.metrics["paper_Et_cm"] = paper_Et_cm
    cmd.metrics["paper_Et_root_cm"] = paper_Et_root_cm
    cmd.metrics["paper_Ej_cm"] = paper_Ej_cm
    cmd.metrics["paper_Eft_cm"] = paper_Eft_cm

    # Per-frame flag: 1 if every non-static body's pos/rot error is below the
    # Per-side breakdown gates for W&B visibility.
    # `Ej_pass_*`  : per-frame E_j < 8 cm gate (joint / body-position error).
    # `Eft_pass_*` : per-frame E_ft < 6 cm gate (fingertip-position error).
    # `Ej_and_Eft_pass_*` : composite (both gates pass on that side).
    ej_pass_l = (left_Ej_cm < 8.0).float()
    ej_pass_r = (right_Ej_cm < 8.0).float()
    eft_pass_l = (left_Eft_cm < 6.0).float()
    eft_pass_r = (right_Eft_cm < 6.0).float()
    ej_and_eft_l = ej_pass_l * eft_pass_l
    ej_and_eft_r = ej_pass_r * eft_pass_r
    cmd.metrics["maniptrans_Ej_pass_left"] = ej_pass_l
    cmd.metrics["maniptrans_Ej_pass_right"] = ej_pass_r
    cmd.metrics["maniptrans_Ej_pass_both"] = ej_pass_l * ej_pass_r
    cmd.metrics["maniptrans_Eft_pass_left"] = eft_pass_l
    cmd.metrics["maniptrans_Eft_pass_right"] = eft_pass_r
    cmd.metrics["maniptrans_Eft_pass_both"] = eft_pass_l * eft_pass_r
    cmd.metrics["maniptrans_Ej_and_Eft_pass_left"] = ej_and_eft_l
    cmd.metrics["maniptrans_Ej_and_Eft_pass_right"] = ej_and_eft_r
    cmd.metrics["maniptrans_Ej_and_Eft_pass_both"] = ej_and_eft_l * ej_and_eft_r
