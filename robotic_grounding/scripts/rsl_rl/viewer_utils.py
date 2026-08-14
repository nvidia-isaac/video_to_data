"""Shared viewer utilities for RSL-RL training/eval scripts."""

from __future__ import annotations


def autoframe_viewer(env_cfg, motion_file: str) -> None:
    """Set viewer eye/lookat from the motion file's actual bounding box.

    Reads object_body_position and robot_{side}_wrist_position from the parquet,
    computes the scene centroid + extent, and positions the camera at a
    135°-azimuth / ~30°-elevation offset with a 6 m minimum distance so the
    cloned env grid stays visible in training video. Falls back silently if the
    parquet is missing or the required fields are absent.
    """
    import logging

    import numpy as np
    import pyarrow.parquet as pq
    from isaaclab.envs import ManagerBasedRLEnvCfg

    logger = logging.getLogger(__name__)

    if not (isinstance(env_cfg, ManagerBasedRLEnvCfg) and hasattr(env_cfg, "viewer")):
        return
    try:
        data = pq.read_table(motion_file).to_pydict()
        pts = []
        obj = data.get("object_body_position", [None])[0]
        if obj:
            pts.append(np.asarray(obj).reshape(-1, 3))
        for side in ("right", "left"):
            wrist = data.get(f"robot_{side}_wrist_position", [None])[0]
            if wrist:
                pts.append(np.asarray(wrist).reshape(-1, 3))
        if not pts:
            return
        all_pts = np.concatenate(pts, axis=0)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        center = 0.5 * (lo + hi)
        extent = max(float(np.linalg.norm(hi - lo)), 0.3)
        dist = max(2.5 * extent, 6.0)
        eye = center + dist * np.array([-0.60, 0.60, 0.57])
        env_cfg.viewer.lookat = tuple(float(c) for c in center)
        env_cfg.viewer.eye = tuple(float(c) for c in eye)
        logger.info(
            f"viewer autoframe: lookat={env_cfg.viewer.lookat}, eye={env_cfg.viewer.eye}"
        )
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning(f"viewer autoframe failed: {e}")
