# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclasses shared between the reconstruction service and tab.

The reconstruction chain is the 16-stage Full (ego_e2e) pipeline driven by
`reconstruction/modules/v2d_pipelines/run_v2d_ego_e2e.py`. The earlier
"Object" 4-stage chain has been retired — its outputs are a strict subset
of Full mode's, just under different filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# 16-stage chain. Mirrors the `_step("<label>", …)` calls in
# `reconstruction/modules/v2d_pipelines/run_v2d_ego_e2e.py`.
STAGES: tuple[str, ...] = (
    "ego_hand_recon",
    "convert_dynhamr_depth",
    "extract_frames",
    "moge_depth",
    "stabilize_intrinsics",
    "grounding_dino",
    "sam2_masks",
    "sam3d_mesh",
    "apply_sam3d_transform",
    "fp_scale",
    "fp_tracking",
    "ekf_smoothing",
    "render_smoothed",
    "hand_alignment",
    "render_aligned",
    "render_unaligned",
)
STAGE_LABELS: dict[str, str] = {
    "ego_hand_recon": "Ego hand reconstruction (ViPE + Dyn-HaMR)",
    "convert_dynhamr_depth": "Convert DynHaMR depth",
    "extract_frames": "Extract frames",
    "moge_depth": "MoGe depth + intrinsics",
    "stabilize_intrinsics": "Stabilise intrinsics",
    "grounding_dino": "Grounding DINO",
    "sam2_masks": "SAM2 mask tracking",
    "sam3d_mesh": "SAM3D mesh generation",
    "apply_sam3d_transform": "Apply SAM3D transform",
    "fp_scale": "Scale estimation",
    "fp_tracking": "FoundationPose tracking",
    "ekf_smoothing": "EKF pose smoothing",
    "render_smoothed": "Render smoothed poses",
    "hand_alignment": "Hand/object depth alignment",
    "render_aligned": "Render aligned (2x2 grid)",
    "render_unaligned": "Render unaligned (comparison)",
}
# Ordered list of (label_prefix, stage_id). The orchestrator's labels include
# a `({depth_source})` suffix on some stages (e.g. "SAM3D mesh generation
# (moge depth)"), so we match by `startswith`. Entries are in the same order
# as STAGES so the parser can also assert forward progress.
STAGE_MARKER: tuple[tuple[str, str], ...] = (
    ("Ego hand reconstruction", "ego_hand_recon"),
    ("Convert DynHaMR depth", "convert_dynhamr_depth"),
    ("Extract frames", "extract_frames"),
    ("MoGe depth", "moge_depth"),
    ("Stabilise intrinsics", "stabilize_intrinsics"),
    ("Grounding DINO", "grounding_dino"),
    ("SAM2 mask tracking", "sam2_masks"),
    ("SAM3D mesh generation", "sam3d_mesh"),
    ("Apply SAM3D transform", "apply_sam3d_transform"),
    ("Scale estimation", "fp_scale"),
    ("FoundationPose tracking", "fp_tracking"),
    ("EKF pose smoothing", "ekf_smoothing"),
    ("Render smoothed poses", "render_smoothed"),
    ("Hand/object depth alignment", "hand_alignment"),
    ("Render aligned", "render_aligned"),
    ("Render unaligned", "render_unaligned"),
)

StageStatus = Literal["pending", "running", "ok", "err"]
DepthSource = Literal["moge", "vipe"]


@dataclass
class ReconstructionRequest:
    """A single segment selected by the user for reconstruction."""

    segment_id: str
    video_path: Path
    start_t: float
    end_t: float
    object_label: str
    action_label: str = ""
    description: str = ""
    ref_frame: int = 0
    object_id: int = 1
    simplify_factor: float = 0.5
    depth_source: DepthSource = "moge"

    @classmethod
    def from_clip_dict(cls, clip: dict) -> ReconstructionRequest:
        """Build a request from one entry of `clips_final.jsonl` or a webapp
        query result clip.

        The two sources use slightly different field names:
          - `clips_final.jsonl`: clip_id, start_t, end_t
          - QueryResult.clips:   (no clip_id), start_time, end_time

        Always prefix the source-video stem onto the segment_id so that
        outputs from clips of different videos can't collide in the
        reconstruction `out_root`. Ingestion mints `clip_id` per-video
        (e.g. "clip_0001"), which isn't unique across runs/videos.
        """
        video_path = Path(clip["video_path"])
        start_t = float(clip.get("start_t", clip.get("start_time", 0.0)))
        end_t = float(clip.get("end_t", clip.get("end_time", 0.0)))
        clip_id = clip.get("clip_id")
        stem = video_path.stem
        if clip_id:
            # Idempotence: don't double-prefix if the clip_id already carries
            # the video stem (some dataset conventions bake it in).
            segment_id = clip_id if clip_id.startswith(f"{stem}_") else f"{stem}__{clip_id}"
        else:
            segment_id = f"{stem}_{start_t:.2f}s-{end_t:.2f}s".replace(".", "_")
        return cls(
            segment_id=segment_id,
            video_path=video_path,
            start_t=start_t,
            end_t=end_t,
            object_label=clip.get("object", ""),
            action_label=clip.get("action", ""),
            description=clip.get("description", ""),
        )


@dataclass
class StageEvent:
    """One progress tick from the reconstruction chain."""

    stage: str
    status: StageStatus
    message: str = ""
    elapsed_s: float = 0.0
    artifact_path: Path | None = None


@dataclass
class ReconstructionResult:
    """Snapshot of artifacts currently on disk for a segment."""

    request: ReconstructionRequest
    seg_dir: Path

    @property
    def render_mp4(self) -> Path | None:
        path = self.seg_dir / f"render_aligned_{self.request.depth_source}.mp4"
        return path if path.is_file() else None

    @property
    def scaled_mesh(self) -> Path | None:
        path = self.seg_dir / f"mesh_scaled_{self.request.depth_source}.obj"
        return path if path.is_file() else None
