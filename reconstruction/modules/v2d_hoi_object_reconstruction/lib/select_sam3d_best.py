# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Suggest the best SAM3D mesh candidate from a multi-frame run.

The selector ranks per-frame SRT results using projection loss as the primary
signal, with small sanity penalties for suspicious scale, anisotropy, missing
debug artifacts, and source frames outside Stage 1.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Optional


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _median(values: list[float]) -> Optional[float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _scale_stats(scale: Any) -> tuple[Optional[float], Optional[float], list[float]]:
    if isinstance(scale, list):
        values = [_finite_float(v) for v in scale]
        vals = [v for v in values if v is not None and v > 0]
    else:
        value = _finite_float(scale)
        vals = [value] if value is not None and value > 0 else []
    if not vals:
        return None, None, []
    log_mean = sum(math.log(v) for v in vals) / len(vals)
    scale_gmean = math.exp(log_mean)
    anisotropy = max(vals) / min(vals)
    return scale_gmean, anisotropy, vals


def _read_selected_frames(sam3d_dir: Path) -> list[str]:
    selected_path = sam3d_dir / "selected_frames.json"
    if selected_path.exists():
        with open(selected_path) as f:
            frames = json.load(f)
        return [str(frame) for frame in frames]
    return sorted(p.name for p in sam3d_dir.iterdir() if p.is_dir() and p.name.isdigit())


def _read_stage1_end(job_dir: Path) -> Optional[int]:
    result_path = job_dir / "stage1_detect_debug" / "result.json"
    if not result_path.exists():
        return None
    try:
        with open(result_path) as f:
            stage1_end = json.load(f).get("stage1_end_frame")
        return int(stage1_end) if stage1_end is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _relative(job_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(job_dir))
    except ValueError:
        return str(path)


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def _load_candidate(job_dir: Path, frame_id: str, stage1_end_frame: Optional[int]) -> dict[str, Any]:
    frame_dir = job_dir / "sam3d" / frame_id
    srt_dir = frame_dir / "srt"
    srt_result_path = srt_dir / "srt_result.json"
    output_glb = srt_dir / "output_scaled.glb"
    render_video = frame_dir / "render_video.mp4"
    render_debug = frame_dir / "render_debug.jpg"
    source_mesh = frame_dir / "mesh.glb"

    candidate: dict[str, Any] = {
        "frame_id": frame_id,
        "valid": False,
        "warnings": [],
        "artifacts": {
            "output_scaled_glb": _relative(job_dir, output_glb),
            "srt_result_json": _relative(job_dir, srt_result_path),
            "render_debug_jpg": _relative(job_dir, render_debug),
            "render_video_mp4": _relative(job_dir, render_video),
        },
    }

    if not srt_result_path.exists():
        candidate["warnings"].append("missing_srt_result")
        return candidate
    if not output_glb.exists():
        candidate["warnings"].append("missing_output_scaled_glb")
        return candidate

    try:
        with open(srt_result_path) as f:
            result = json.load(f)
    except json.JSONDecodeError:
        candidate["warnings"].append("invalid_srt_result_json")
        return candidate

    loss = _finite_float(result.get("total_loss"))
    if loss is None:
        loss = _finite_float(result.get("best_tracked_loss"))
    if loss is None:
        candidate["warnings"].append("missing_finite_srt_loss")
        return candidate

    scale_gmean, anisotropy, scale_values = _scale_stats(result.get("scale"))
    if scale_gmean is None or anisotropy is None:
        candidate["warnings"].append("missing_valid_scale")

    try:
        mesh_bytes = output_glb.stat().st_size
    except OSError:
        mesh_bytes = 0
    try:
        source_mesh_bytes = source_mesh.stat().st_size
    except OSError:
        source_mesh_bytes = 0

    frame_int = int(frame_id) if frame_id.isdigit() else None
    is_stage1_source = (
        stage1_end_frame is None
        or frame_int is None
        or frame_int <= stage1_end_frame
    )
    if not is_stage1_source:
        candidate["warnings"].append("source_frame_after_stage1")
    if not render_video.exists():
        candidate["warnings"].append("missing_render_video")
    if not render_debug.exists():
        candidate["warnings"].append("missing_render_debug")
    if result.get("optimizer_success") is False:
        candidate["warnings"].append("optimizer_reported_not_successful")

    candidate.update({
        "valid": True,
        "total_loss": loss,
        "scale": result.get("scale"),
        "scale_geometric_mean": scale_gmean,
        "scale_values": scale_values,
        "anisotropy": anisotropy,
        "mesh_bytes": mesh_bytes,
        "source_mesh_bytes": source_mesh_bytes,
        "num_views": result.get("num_views"),
        "optimizer_success": result.get("optimizer_success"),
        "srt_strategy": result.get("srt_strategy"),
        "orientation_key": result.get("orientation_key"),
        "orientation_source": result.get("orientation_source"),
        "is_stage1_source": is_stage1_source,
    })
    return candidate


def _score_candidates(candidates: list[dict[str, Any]]) -> None:
    valid = [c for c in candidates if c.get("valid")]
    median_loss = _median([float(c["total_loss"]) for c in valid])
    median_scale = _median([
        float(c["scale_geometric_mean"])
        for c in valid
        if c.get("scale_geometric_mean") is not None
    ])
    median_mesh_bytes = _median([
        float(c["mesh_bytes"])
        for c in valid
        if c.get("mesh_bytes")
    ])
    reference_loss = median_loss if median_loss and median_loss > 0 else 1.0

    for c in candidates:
        if not c.get("valid"):
            c["score"] = None
            continue
        score = float(c["total_loss"])
        penalties: dict[str, float] = {}

        scale_gmean = c.get("scale_geometric_mean")
        if median_scale and scale_gmean:
            scale_penalty = reference_loss * 0.15 * abs(math.log(float(scale_gmean) / median_scale))
            if scale_penalty:
                penalties["relative_scale"] = scale_penalty
                score += scale_penalty

        anisotropy = c.get("anisotropy")
        if anisotropy is not None and anisotropy > 1.5:
            anisotropy_penalty = reference_loss * 0.20 * (float(anisotropy) - 1.5)
            penalties["anisotropy"] = anisotropy_penalty
            score += anisotropy_penalty

        mesh_bytes = c.get("mesh_bytes") or 0
        if median_mesh_bytes and mesh_bytes < 0.5 * median_mesh_bytes:
            mesh_penalty = reference_loss * 0.10 * (1.0 - float(mesh_bytes) / median_mesh_bytes)
            penalties["small_output_mesh"] = mesh_penalty
            score += mesh_penalty

        if not c.get("is_stage1_source", True):
            stage_penalty = reference_loss * 0.25
            penalties["source_frame_after_stage1"] = stage_penalty
            score += stage_penalty

        if c.get("optimizer_success") is False:
            opt_penalty = reference_loss * 0.05
            penalties["optimizer_reported_not_successful"] = opt_penalty
            score += opt_penalty

        if "missing_render_video" in c.get("warnings", []):
            video_penalty = reference_loss * 0.05
            penalties["missing_render_video"] = video_penalty
            score += video_penalty

        c["score"] = score
        c["score_penalties"] = penalties


def select_best_sam3d_frame(job_dir: Path, output_dir: Optional[Path] = None) -> dict[str, Any]:
    """Rank SAM3D SRT candidates and copy the suggested best artifacts."""
    job_dir = Path(job_dir)
    sam3d_dir = job_dir / "sam3d"
    if not sam3d_dir.is_dir():
        raise FileNotFoundError(f"SAM3D directory not found: {sam3d_dir}")

    output_dir = Path(output_dir) if output_dir is not None else sam3d_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = _read_selected_frames(sam3d_dir)
    stage1_end_frame = _read_stage1_end(job_dir)
    candidates = [_load_candidate(job_dir, frame_id, stage1_end_frame) for frame_id in frames]
    _score_candidates(candidates)
    ranked = sorted(
        candidates,
        key=lambda c: (c.get("score") is None, c.get("score") if c.get("score") is not None else float("inf"), c["frame_id"]),
    )
    valid_ranked = [c for c in ranked if c.get("valid")]
    if not valid_ranked:
        summary = {
            "best_frame": None,
            "stage1_end_frame": stage1_end_frame,
            "selection_method": "srt_total_loss_with_sanity_penalties",
            "ranked_frames": ranked,
            "note": "No valid frame had both srt_result.json and output_scaled.glb.",
        }
        (output_dir / "best_frame.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise ValueError("No valid SAM3D SRT candidates found")

    best = valid_ranked[0]
    frame_id = best["frame_id"]
    frame_dir = sam3d_dir / frame_id
    copied = {
        "output_scaled_glb": _copy_if_exists(frame_dir / "srt" / "output_scaled.glb", output_dir / "output_scaled.glb"),
        "srt_result_json": _copy_if_exists(frame_dir / "srt" / "srt_result.json", output_dir / "srt_result.json"),
        "render_debug_jpg": _copy_if_exists(frame_dir / "render_debug.jpg", output_dir / "render_debug.jpg"),
        "render_video_mp4": _copy_if_exists(frame_dir / "render_video.mp4", output_dir / "render_video.mp4"),
        "source_mesh_glb": _copy_if_exists(frame_dir / "mesh.glb", output_dir / "source_mesh.glb"),
        "transform_json": _copy_if_exists(frame_dir / "transform.json", output_dir / "transform.json"),
        "intrinsics_json": _copy_if_exists(frame_dir / "intrinsics.json", output_dir / "intrinsics.json"),
    }
    (output_dir / "best_frame.txt").write_text(frame_id + "\n")

    summary = {
        "best_frame": frame_id,
        "stage1_end_frame": stage1_end_frame,
        "selection_method": "lowest SRT total_loss plus small sanity penalties; suggested only, inspect render_video",
        "score_lower_is_better": True,
        "best_artifacts": {
            key: _relative(job_dir, output_dir / name)
            for key, name in copied.items()
            if name is not None
        },
        "ranked_frames": ranked,
    }
    (output_dir / "best_frame.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    summary = select_best_sam3d_frame(args.job_dir, args.output_dir)
    print(json.dumps({
        "best_frame": summary["best_frame"],
        "best_artifacts": summary["best_artifacts"],
        "selection_method": summary["selection_method"],
    }, indent=2))


if __name__ == "__main__":
    main()
