"""Resolve v2d BundleSDF runtime configuration.

BundleSDF consumes concrete numeric values. This module keeps user-facing policy
in the YAML and converts it before NVBundleSDF is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ObjectScaleEstimate:
    diagonal_m: float
    extent_short_m: float
    extent_mid_m: float
    extent_long_m: float
    aspect_ratio: float
    mask_aspect_ratio: float


def resolve_bundlesdf_config(
    config: dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Resolve v2d-specific BundleSDF config policy into concrete values."""
    nerf = config.setdefault("nerf", {})
    texture_bake = config.setdefault("texture_bake", {})
    auto_tune = config.get("auto_tune", {}) or {}

    # Newer upstream releases require these keys even when USD export is not
    # used. Preserve the previous V2D behavior unless explicitly enabled.
    texture_bake.setdefault("export_usd", False)
    texture_bake.setdefault("export_usdz", False)

    _resolve_texture_zfar(texture_bake, nerf, logger)
    _resolve_sdf_far_policy(config, output_path, auto_tune, logger)
    _resolve_sdf_resolution_policy(config, output_path, auto_tune, logger)

    return config


def dump_resolved_config(config: dict[str, Any], path: Path) -> None:
    """Write a YAML-safe copy of the effective config."""
    with open(path, "w") as f:
        yaml.safe_dump(_to_yaml_safe(config), f, sort_keys=False)


def _resolve_texture_zfar(
    texture_bake: dict[str, Any],
    nerf: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Copy texture zfar policy into the NeRF config visible to NVIDIA code."""
    zfar_cfg = texture_bake.get("zfar")
    if zfar_cfg is None:
        return

    if isinstance(zfar_cfg, (int, float)):
        nerf["texture_zfar"] = float(zfar_cfg)
        logger.info("Texture zfar: using explicit renderer zfar %.3f", nerf["texture_zfar"])
        return

    if not isinstance(zfar_cfg, dict):
        raise ValueError("texture_bake.zfar must be a number or mapping")

    mode = str(zfar_cfg.get("mode", "auto")).lower()
    if mode in {"off", "disabled", "none"}:
        nerf.pop("texture_zfar", None)
        nerf.pop("texture_zfar_camera_distance_margin", None)
        logger.info("Texture zfar: config override disabled; upstream fallback will be used")
        return

    if mode in {"explicit", "fixed"}:
        if "value" not in zfar_cfg:
            raise ValueError("texture_bake.zfar.value is required when mode is explicit")
        nerf["texture_zfar"] = float(zfar_cfg["value"])
        logger.info("Texture zfar: using explicit renderer zfar %.3f", nerf["texture_zfar"])
        return

    if mode != "auto":
        raise ValueError(f"Unsupported texture_bake.zfar.mode: {mode}")

    margin = float(zfar_cfg.get("camera_distance_margin", 2.0))
    if margin <= 0:
        raise ValueError("texture_bake.zfar.camera_distance_margin must be > 0")
    nerf["texture_zfar_camera_distance_margin"] = margin
    nerf.pop("texture_zfar", None)
    logger.info("Texture zfar: auto with camera_distance_margin=%.3f", margin)


def _resolve_sdf_far_policy(
    config: dict[str, Any],
    output_path: Path,
    auto_tune: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Resolve nerf.far from masked object depth.

    nerf.far is a training-time depth range in meters. If it is too small,
    BundleSDF drops valid object surface depths from SDF supervision and mesh
    extraction can fail with no zero crossing.
    """
    nerf = config.setdefault("nerf", {})
    original_far = nerf.get("far")
    far_policy = auto_tune.get("far", {}) or {}

    if not _policy_enabled(far_policy, original_far):
        return

    percentile = float(far_policy.get("depth_percentile", 99.5))
    margin = float(far_policy.get("margin", 1.25))
    min_m = float(far_policy.get("min_m", 1.0))
    max_m = float(far_policy.get("max_m", 8.0))
    max_frames = int(far_policy.get("max_frames", 16))

    if not (0.0 < percentile <= 100.0):
        raise ValueError("auto_tune.far.depth_percentile must be in (0, 100]")
    if margin <= 0:
        raise ValueError("auto_tune.far.margin must be > 0")
    if max_frames <= 0:
        raise ValueError("auto_tune.far.max_frames must be > 0")

    depth_far = _estimate_masked_depth_far_m(
        output_path,
        depth_percentile=percentile,
        max_frames=max_frames,
        logger=logger,
    )
    if depth_far is None:
        if _is_auto(original_far):
            fallback = float(far_policy.get("fallback_m", min_m))
            nerf["far"] = _clamp(fallback, min_m, max_m)
            logger.warning(
                "Could not estimate masked object depth; using fallback nerf.far=%.3fm",
                nerf["far"],
            )
        else:
            logger.warning("Could not estimate masked object depth; keeping configured nerf.far=%s", original_far)
        return

    far = _clamp(depth_far * margin, min_m, max_m)
    nerf["far"] = far
    logger.info(
        "Resolved nerf.far=%.3fm from masked depth p%.2f=%.3fm, margin=%.3f",
        far,
        percentile,
        depth_far,
        margin,
    )


def _resolve_sdf_resolution_policy(
    config: dict[str, Any],
    output_path: Path,
    auto_tune: dict[str, Any],
    logger: logging.Logger,
) -> None:
    nerf = config.setdefault("nerf", {})
    original_trunc = nerf.get("trunc")
    original_trunc_start = nerf.get("trunc_start")

    trunc_policy = auto_tune.get("trunc", {}) or {}
    mesh_policy = auto_tune.get("mesh_resolution", {}) or {}
    resolve_trunc = _policy_enabled(trunc_policy, original_trunc)
    resolve_mesh = _policy_enabled(mesh_policy, nerf.get("mesh_resolution"))

    if not resolve_trunc and not resolve_mesh:
        return

    object_scale = _estimate_object_scale_m(config, output_path, logger)
    if object_scale is None:
        logger.warning(
            "Could not estimate object scale; keeping configured trunc=%s and mesh_resolution=%s",
            nerf.get("trunc"),
            nerf.get("mesh_resolution"),
        )
        return

    tuning_diameter, slender_adjusted = _resolve_tuning_diameter(object_scale, trunc_policy)
    logger.info(
        "Estimated object diameter for BundleSDF tuning: %.4fm "
        "(effective=%.4fm, extents short/mid/long=%.4f/%.4f/%.4fm, "
        "3d_aspect=%.2f, mask_aspect_p90=%.2f, slender_adjusted=%s)",
        object_scale.diagonal_m,
        tuning_diameter,
        object_scale.extent_short_m,
        object_scale.extent_mid_m,
        object_scale.extent_long_m,
        object_scale.aspect_ratio,
        object_scale.mask_aspect_ratio,
        slender_adjusted,
    )

    if resolve_trunc:
        trunc = _clamp(
            tuning_diameter * float(trunc_policy.get("diameter_ratio", 0.01)),
            float(trunc_policy.get("min_m", 0.005)),
            float(trunc_policy.get("max_m", 0.03)),
        )
        nerf["trunc"] = trunc
        if _is_auto(original_trunc_start) or original_trunc_start == original_trunc:
            nerf["trunc_start"] = trunc
        logger.info("Resolved nerf.trunc=%.4f, nerf.trunc_start=%s", trunc, nerf.get("trunc_start"))

    if resolve_mesh:
        trunc_value = float(nerf["trunc"])
        mesh_resolution = _clamp(
            trunc_value * float(mesh_policy.get("trunc_ratio", 0.5)),
            float(mesh_policy.get("min_m", 0.002)),
            float(mesh_policy.get("max_m", 0.008)),
        )
        mesh_resolution = min(mesh_resolution, trunc_value)
        nerf["mesh_resolution"] = mesh_resolution
        logger.info("Resolved nerf.mesh_resolution=%.4f", mesh_resolution)


def _policy_enabled(policy: dict[str, Any], current_value: Any) -> bool:
    if _is_auto(current_value):
        return True
    return bool(policy.get("enabled", False))


def _resolve_tuning_diameter(
    object_scale: ObjectScaleEstimate,
    trunc_policy: dict[str, Any],
) -> tuple[float, bool]:
    """Return the object scale used by auto-trunc.

    The raw bbox diagonal is a good size proxy for bulky objects, but it
    overestimates long/thin tools because most of the diagonal is just length.
    For elongated objects, cap the effective diameter by multiples of the
    middle and short extents so the SDF truncation remains tied to thickness/width.
    """
    tuning_diameter = object_scale.diagonal_m
    slender_cfg = trunc_policy.get("slender", {}) or {}
    if not bool(slender_cfg.get("enabled", True)):
        return tuning_diameter, False

    aspect_threshold = float(slender_cfg.get("aspect_ratio_threshold", 2.5))
    mid_extent_multiplier = float(slender_cfg.get("mid_extent_multiplier", 3.0))
    short_extent_multiplier = float(slender_cfg.get("short_extent_multiplier", 2.0))
    if aspect_threshold <= 1.0:
        raise ValueError("auto_tune.trunc.slender.aspect_ratio_threshold must be > 1")
    if mid_extent_multiplier <= 0:
        raise ValueError("auto_tune.trunc.slender.mid_extent_multiplier must be > 0")
    if short_extent_multiplier <= 0:
        raise ValueError("auto_tune.trunc.slender.short_extent_multiplier must be > 0")

    shape_aspect = max(object_scale.aspect_ratio, object_scale.mask_aspect_ratio)
    if shape_aspect < aspect_threshold:
        return tuning_diameter, False

    slender_cap = min(
        object_scale.extent_mid_m * mid_extent_multiplier,
        object_scale.extent_short_m * short_extent_multiplier,
    )
    if not np.isfinite(slender_cap) or slender_cap <= 0:
        return tuning_diameter, False

    adjusted = min(tuning_diameter, float(slender_cap))
    return adjusted, adjusted < tuning_diameter


def _estimate_object_scale_m(
    config: dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> ObjectScaleEstimate | None:
    intrinsic = np.asarray(config["camera_config"]["intrinsic"], dtype=np.float32).reshape(3, 3)
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

    depth_dir = output_path / "depth"
    mask_dir = output_path / "masks"
    depth_files = sorted(list(depth_dir.glob("*.npy")) + list(depth_dir.glob("*.png")))
    if not depth_files:
        return None

    mask_aspect_ratio = _estimate_mask_bbox_aspect_ratio(mask_dir)
    diameters: list[float] = []
    sorted_extents: list[np.ndarray] = []
    step = max(1, len(depth_files) // 8)
    for depth_path in depth_files[::step]:
        mask_path = _matching_mask_path(mask_dir, depth_path.stem)
        if mask_path is None:
            continue

        depth = _load_depth_m(depth_path)
        mask = _load_mask(mask_path)
        if depth is None or mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
        if valid.sum() < 100:
            continue

        ys, xs = np.nonzero(valid)
        if len(xs) > 20000:
            ids = np.linspace(0, len(xs) - 1, 20000).astype(np.int64)
            xs = xs[ids]
            ys = ys[ids]
        z = depth[ys, xs]
        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)
        lo = np.percentile(pts, 2, axis=0)
        hi = np.percentile(pts, 98, axis=0)
        extent = hi - lo
        diameter = float(np.linalg.norm(extent))
        if np.isfinite(diameter) and diameter > 0:
            diameters.append(diameter)
            sorted_extents.append(np.sort(extent.astype(np.float32)))

    if not diameters or not sorted_extents:
        logger.warning("Object scale estimation found no valid masked depth samples")
        return None

    median_extent = np.median(np.stack(sorted_extents, axis=0), axis=0)
    extent_short, extent_mid, extent_long = [float(v) for v in median_extent]
    aspect_ratio = extent_long / max(extent_mid, 1e-6)
    return ObjectScaleEstimate(
        diagonal_m=float(np.median(diameters)),
        extent_short_m=extent_short,
        extent_mid_m=extent_mid,
        extent_long_m=extent_long,
        aspect_ratio=float(aspect_ratio),
        mask_aspect_ratio=mask_aspect_ratio,
    )


def _estimate_mask_bbox_aspect_ratio(mask_dir: Path) -> float:
    mask_files = sorted(
        list(mask_dir.glob("*.png"))
        + list(mask_dir.glob("*.npy"))
        + list(mask_dir.glob("*.jpg"))
        + list(mask_dir.glob("*.jpeg"))
    )
    aspect_ratios: list[float] = []
    for mask_path in mask_files:
        mask = _load_mask(mask_path)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        valid = mask > 0
        if valid.sum() < 100:
            continue
        ys, xs = np.nonzero(valid)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        aspect_ratios.append(max(width, height) / max(1, min(width, height)))

    if not aspect_ratios:
        return 1.0
    return float(np.percentile(aspect_ratios, 90))


def _estimate_masked_depth_far_m(
    output_path: Path,
    depth_percentile: float,
    max_frames: int,
    logger: logging.Logger,
) -> float | None:
    depth_dir = output_path / "depth"
    mask_dir = output_path / "masks"
    depth_files = sorted(list(depth_dir.glob("*.npy")) + list(depth_dir.glob("*.png")))
    if not depth_files:
        return None

    frame_depths: list[float] = []
    step = max(1, len(depth_files) // max_frames)
    for depth_path in depth_files[::step]:
        mask_path = _matching_mask_path(mask_dir, depth_path.stem)
        if mask_path is None:
            continue

        depth = _load_depth_m(depth_path)
        mask = _load_mask(mask_path)
        if depth is None or mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
        if valid.sum() < 100:
            continue

        frame_depth = float(np.percentile(depth[valid], depth_percentile))
        if np.isfinite(frame_depth) and frame_depth > 0:
            frame_depths.append(frame_depth)

    if not frame_depths:
        logger.warning("nerf.far auto tuning found no valid masked depth samples")
        return None

    # Use the farthest sampled-frame percentile so the SDF valid-depth range
    # covers the object throughout the scan, while still rejecting extreme noise.
    return max(frame_depths)


def _load_depth_m(path: Path) -> np.ndarray | None:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    depth = np.zeros_like(raw, dtype=np.float32)
    valid = raw > 0
    depth[valid] = 1.0 / (raw[valid] / 65535.0) - 1.0
    return depth


def _load_mask(path: Path) -> np.ndarray | None:
    if path.suffix == ".npy":
        return np.load(path)
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _matching_mask_path(mask_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".npy", ".jpg", ".jpeg"):
        path = mask_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _clamp(value: float, lower: float, upper: float) -> float:
    if lower <= 0 or upper <= 0 or lower > upper:
        raise ValueError(f"Invalid clamp range: min={lower}, max={upper}")
    return min(max(value, lower), upper)


def _to_yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_yaml_safe(v) for v in value]
    return value
