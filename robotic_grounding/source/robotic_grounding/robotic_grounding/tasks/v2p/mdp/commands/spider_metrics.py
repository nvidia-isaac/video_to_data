"""SPIDER-paper-exact object tracking metrics for DualHandsObjectTrackingCommand.

Separated from hand_object_commands.py so all SPIDER-specific logic lives here.
All public functions take ``cmd`` (the command term instance) as their first
argument and operate on its attributes in-place.

Reference: SPIDER postprocess/get_success_rate.py (compute_object_tracking_error).
"""

from __future__ import annotations

from typing import Any

import torch


def register_spider_metric_keys(cmd: Any) -> None:
    """Register SPIDER metric keys and running buffers on *cmd*.

    Sets the following in ``cmd.metrics`` and as direct attributes:

    - ``cmd.metrics["spider_obj_pos_err"]``
    - ``cmd.metrics["spider_obj_rot_err"]``
    - ``cmd.metrics["spider_pos_within_threshold"]``
    - ``cmd.metrics["spider_rot_within_threshold"]``
    - ``cmd.metrics["spider_both_within_threshold"]``
    - ``cmd._spider_sim_pos_sum``   : (E, B, 3) running position accumulator
    - ``cmd._spider_step_count``    : (E,) step counter, reset per-env in _resample_command

    Called once from ``DualHandsObjectTrackingCommand._init_metrics()``.
    """
    z = torch.zeros(cmd.num_envs, device=cmd.device)
    cmd.metrics["spider_obj_pos_err"] = z.clone()
    cmd.metrics["spider_obj_rot_err"] = z.clone()
    cmd.metrics["spider_pos_within_threshold"] = z.clone()
    cmd.metrics["spider_rot_within_threshold"] = z.clone()
    cmd.metrics["spider_both_within_threshold"] = z.clone()
    cmd._spider_sim_pos_sum = torch.zeros(
        cmd.num_envs, cmd.num_bodies, 3, device=cmd.device
    )
    cmd._spider_step_count = torch.zeros(cmd.num_envs, device=cmd.device)


def update_spider_metrics(cmd: Any) -> dict[str, torch.Tensor]:
    """Compute SPIDER-paper object tracking errors and write them to ``cmd.metrics``.

    Mirrors ``compute_object_tracking_error`` in
    spider/postprocess/get_success_rate.py:118-236:

    1. Accumulate running sim-position sum to estimate the episode's sim_pos_mean
       (exact at episode end; running approximation during the episode).
    2. Mean-subtract sim and ref position trajectories.
    3. ``||sim_dev - ref_dev||`` per body, masked to exclude static bodies.
    4. Average over non-static bodies → one scalar per env per step.
    Rotation error uses no mean subtraction.

    Called once per step from ``DualHandsObjectTrackingCommand._update_metrics()``.

    Returns a dict of intermediate tensors shared with
    :func:`maniptrans_metrics.update_maniptrans_metrics`:

    - ``cur_sim_pos``      : ``(E, B, 3)``
    - ``cur_ref_pos``      : ``(E, B, 3)``
    - ``rot_err_per_body`` : ``(E, B)``
    - ``non_static``       : ``(1, B)`` float mask (1 = non-static)
    - ``n_non_static``     : ``(E,)`` non-static body count (≥ 1)
    - ``spider_rot_err``   : ``(E,)`` mean rotation error over non-static bodies
    """
    import isaaclab.utils.math as math_utils  # noqa: PLC0415

    cur_sim_pos = cmd.object_position_e  # (E, B, 3)
    cur_ref_pos = cmd.object_body_position_command_e  # (E, B, 3)

    cmd._spider_sim_pos_sum += cur_sim_pos
    cmd._spider_step_count += 1.0
    sim_pos_mean_running = cmd._spider_sim_pos_sum / cmd._spider_step_count.view(
        -1, 1, 1
    )

    sim_dev = cur_sim_pos - sim_pos_mean_running
    ref_dev = cur_ref_pos - cmd._spider_ref_pos_mean_b.unsqueeze(0)
    pos_err_per_body = torch.norm(sim_dev - ref_dev, dim=-1)  # (E, B)

    rot_err_per_body = math_utils.quat_error_magnitude(
        cmd.object_orientation_e,
        cmd.object_body_wxyz_command_e,
    )  # (E, B)

    non_static = (~cmd._spider_static_body_mask).float().unsqueeze(0)  # (1, B)
    n_non_static = non_static.sum(dim=-1).clamp(min=1.0)
    spider_pos_err = (pos_err_per_body * non_static).sum(dim=-1) / n_non_static
    spider_rot_err = (rot_err_per_body * non_static).sum(dim=-1) / n_non_static

    cmd.metrics["spider_obj_pos_err"] = spider_pos_err
    cmd.metrics["spider_obj_rot_err"] = spider_rot_err
    cmd.metrics["spider_pos_within_threshold"] = (spider_pos_err <= 0.1).float()
    cmd.metrics["spider_rot_within_threshold"] = (spider_rot_err <= 0.5).float()
    cmd.metrics["spider_both_within_threshold"] = (
        cmd.metrics["spider_pos_within_threshold"]
        * cmd.metrics["spider_rot_within_threshold"]
    )

    return {
        "cur_sim_pos": cur_sim_pos,
        "cur_ref_pos": cur_ref_pos,
        "rot_err_per_body": rot_err_per_body,
        "non_static": non_static,
        "n_non_static": n_non_static,
        "spider_rot_err": spider_rot_err,
    }
