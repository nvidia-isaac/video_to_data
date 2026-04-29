"""Motion data loading for tracking commands.

Thin shim over `robotic_grounding.motion_schema`: resolves a `cfg.motion_file`
(file or Hive partition directory) into a populated `MotionData`, then
resolves the EE body IDs on the live robot.

The `MotionData` dataclass is re-exported for backward compatibility with
callers that used to import it from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robotic_grounding.motion_schema import (
    SINGLE_ROBOT,
    MotionData,
    load_motion_data_parquet,
)

if TYPE_CHECKING:
    import torch
    from isaaclab.assets import Articulation

    from .tracking_command_cfg import TrackingCommandCfg


__all__ = ["MotionData", "load_motion_data"]


def load_motion_data(
    cfg: TrackingCommandCfg,
    robot: Articulation,
    device: torch.device,
) -> MotionData:
    """Load motion data from a `motion_v1` parquet and resolve robot body IDs.

    Args:
        cfg: The tracking command configuration (uses `cfg.motion_file`).
        robot: The live robot articulation, used to resolve EE body IDs.
        device: Target torch device for tensors.

    Returns:
        A populated `MotionData` with `ee_link_ids` resolved against the robot.

    Raises:
        ValueError: If the loaded file is not a `single_robot` motion. The
            whole-body `TrackingCommand` indexes whole-body joint state and
            cannot consume `dual_hand` files; those should be loaded through
            the dual-hand command term or `replay_data.load_replay_trajectory`.
    """
    md = load_motion_data_parquet(cfg.motion_file, device=str(device))

    if md.motion_kind != SINGLE_ROBOT:
        raise ValueError(
            f"TrackingCommand requires motion_kind={SINGLE_ROBOT!r} but the "
            f"file at {cfg.motion_file!r} has motion_kind={md.motion_kind!r}. "
            f"Dual-hand motions belong to `dual_hands_object_tracking_command` "
            f"or `replay_data.load_replay_trajectory`."
        )

    if md.ee_link_names:
        ee_link_ids, _ = robot.find_bodies(list(md.ee_link_names))
        md.ee_link_ids = ee_link_ids

    return md
