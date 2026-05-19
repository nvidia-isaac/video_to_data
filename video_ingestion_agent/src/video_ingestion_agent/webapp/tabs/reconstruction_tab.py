# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruction tab — runs the 16-stage hand + object reconstruction chain.

User picks a segment from a `clips_final.jsonl` (or types a segment_id), clicks
Run, and watches stage-by-stage progress driven by `[run ]/[skip]` markers from
reconstruction's `run_v2d_ego_e2e.py` orchestrator. Once the aligned render is
on disk it plays inline; the metric mesh opens in `gr.Model3D` for interactive
inspection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from video_ingestion_agent.webapp.config import AppConfig
from video_ingestion_agent.webapp.models.reconstruction import (
    STAGE_LABELS,
    STAGES,
    ReconstructionRequest,
)
from video_ingestion_agent.webapp.services.reconstruction_service import (
    ReconstructionService,
)

logger = logging.getLogger(__name__)


def _read_segments_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL reader; matches reconstruction_interface/_common/ingestion_io.py shape."""
    import json

    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _segment_choices(clips: list[dict]) -> list[str]:
    """Render dropdown choices like '<clip_id>  —  picking up wooden bowl'."""
    out: list[str] = []
    for c in clips:
        seg_id = c.get("clip_id", "?")
        action = c.get("action", "?")
        obj = c.get("object", "?")
        out.append(f"{seg_id}  —  {action} ({obj})")
    return out


def _render_status_html(stage_status: dict[str, str], current_stage: str | None) -> str:
    """Render a compact horizontal status bar for the 16 chain stages."""
    parts = ['<div class="recon-bar">']
    icons = {"pending": "○", "running": "◔", "ok": "●", "err": "✗"}
    for s in STAGES:
        cls = stage_status.get(s, "pending")
        if s == current_stage and cls == "running":
            cls = "running"
        parts.append(
            f'<span class="recon-stage recon-{cls}">{icons.get(cls, "○")} {STAGE_LABELS[s]}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def _initial_status_html() -> str:
    return _render_status_html({}, current_stage=None)


def create_reconstruction_tab(services: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Create the Reconstruction tab. Returns a dict of components for testing."""
    components: dict[str, Any] = {}
    svc: ReconstructionService | None = services.get("reconstruction")

    with gr.Column(elem_classes=["recon-shell"]):
        gr.Markdown(
            "Pick an action segment and watch the 16-stage hand + object "
            "reconstruction chain run end-to-end. Each stage shells out to a "
            "Docker container (`v2d_*:latest`). The full chain takes about "
            "**~10 minutes per segment** depending on clip length."
        )

        # Banner: setup problems (missing weights, unbuilt images, …).
        if svc is None:
            problem_msg = (
                "**Reconstruction is not configured.** Add a `reconstruction:` "
                "block to your webapp config (see `configs/webapp.yaml`)."
            )
        else:
            issues = svc.config.validate()
            problem_msg = (
                "" if not issues else "**Setup issues:**\n\n" + "\n".join(f"- {i}" for i in issues)
            )
        setup_banner = gr.Markdown(value=problem_msg, visible=bool(problem_msg))

        # Segment source.
        with gr.Row():
            segments_path = gr.Textbox(
                label="clips_final.jsonl",
                placeholder="runs/<run>/clips_final.jsonl",
                value="",
                scale=4,
            )
            load_btn = gr.Button("Load segments", variant="secondary", scale=1)

        segment_picker = gr.Dropdown(label="Segment", choices=[], value=None, interactive=True)

        with gr.Row():
            depth_source = gr.Dropdown(
                choices=["moge", "vipe"],
                value="moge",
                label="Depth source",
                scale=1,
            )
            ref_frame = gr.Number(label="Reference frame", value=0, precision=0, scale=1)
            object_id = gr.Number(label="object_id", value=1, precision=0, scale=1)
            simplify_factor = gr.Number(label="Mesh simplify factor", value=0.5, scale=1)
            run_btn = gr.Button(
                "Run reconstruction",
                variant="primary",
                scale=2,
                interactive=svc is not None and not problem_msg,
            )

        status_html = gr.HTML(value=_initial_status_html(), elem_classes=["recon-status"])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Render preview**")
                video_out = gr.Video(label="render", interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("**Metric mesh (interactive)**")
                mesh_out = gr.Model3D(
                    label="mesh",
                    interactive=False,
                    clear_color=[0.05, 0.05, 0.05, 1.0],
                )

        with gr.Accordion("Log", open=False, elem_classes=["recon-log-accordion"]):
            log_box = gr.Textbox(
                label="Stage stdout",
                lines=18,
                max_lines=40,
                interactive=False,
                autoscroll=True,
                elem_classes=["log-output"],
            )

        # Cached segments parsed from segments_path.
        segments_state = gr.State(value=[])

    # ─── handlers ───────────────────────────────────────────────────────

    def on_load_segments(path_str: str) -> tuple:
        path_str = (path_str or "").strip()
        if not path_str:
            return gr.update(choices=[], value=None), [], "Please enter a JSONL path."
        path = Path(path_str)
        if not path.is_file():
            return gr.update(choices=[], value=None), [], f"Not found: {path}"
        try:
            clips = _read_segments_jsonl(path)
        except Exception as e:  # noqa: BLE001
            return gr.update(choices=[], value=None), [], f"Read failed: {e}"
        if not clips:
            return gr.update(choices=[], value=None), [], "No segments in file."
        choices = _segment_choices(clips)
        return (
            gr.update(choices=choices, value=choices[0]),
            clips,
            f"Loaded {len(clips)} segments.",
        )

    def on_run(
        segment_label: str | None,
        clips: list[dict],
        ref_frame_v: float,
        object_id_v: float,
        simplify_v: float,
        depth_source_v: str = "moge",
    ):
        # Initial yield: clear viewers + status.
        yield _initial_status_html(), "", None, None

        if svc is None:
            yield (
                _render_status_html({s: "err" for s in STAGES}, None),
                "Reconstruction service is not configured.",
                None,
                None,
            )
            return

        if not segment_label or not clips:
            yield (
                _render_status_html({s: "err" for s in STAGES}, None),
                "Pick a segment first.",
                None,
                None,
            )
            return

        # Resolve picked label → clip dict (segment_id is the prefix before "  —").
        seg_id = segment_label.split("  —", 1)[0].strip()
        match = next((c for c in clips if c.get("clip_id") == seg_id), None)
        if match is None:
            yield (
                _render_status_html({s: "err" for s in STAGES}, None),
                f"Segment {seg_id!r} not found in loaded clips.",
                None,
                None,
            )
            return

        request = ReconstructionRequest.from_clip_dict(match)
        request.ref_frame = int(ref_frame_v)
        request.object_id = int(object_id_v)
        request.simplify_factor = float(simplify_v)
        request.depth_source = depth_source_v if depth_source_v in ("moge", "vipe") else "moge"  # type: ignore[assignment]

        stage_status: dict[str, str] = {s: "pending" for s in STAGES}
        log_buf: list[str] = [
            f"# segment_id: {request.segment_id}",
            f"# object: {request.object_label}",
            f"# action: {request.action_label}",
            f"# range:  {request.start_t:.2f}s–{request.end_t:.2f}s",
            f"# depth_source: {request.depth_source}",
            "",
        ]

        for ev in svc.run(request):
            # Don't downgrade a completed stage back to "running" via a
            # log-line passthrough event; only update on real transitions.
            prev = stage_status.get(ev.stage, "pending")
            if not (prev == "ok" and ev.status == "running"):
                stage_status[ev.stage] = ev.status
            if ev.message:
                log_buf.append(f"[{ev.stage}] {ev.message}")
                # Cap the buffer so Gradio doesn't bog down.
                if len(log_buf) > 4000:
                    log_buf = log_buf[-3000:]

            current = ev.stage if ev.status == "running" else None
            result = svc.collect_result(request)
            yield (
                _render_status_html(stage_status, current),
                "\n".join(log_buf),
                str(result.render_mp4) if result.render_mp4 else None,
                str(result.scaled_mesh) if result.scaled_mesh else None,
            )

        # Final pass picks up any artifacts written after the last log line.
        result = svc.collect_result(request)
        yield (
            _render_status_html(stage_status, current_stage=None),
            "\n".join(log_buf),
            str(result.render_mp4) if result.render_mp4 else None,
            str(result.scaled_mesh) if result.scaled_mesh else None,
        )

    # ─── wire events ────────────────────────────────────────────────────

    load_btn.click(
        fn=on_load_segments,
        inputs=[segments_path],
        outputs=[segment_picker, segments_state, setup_banner],
    )

    run_btn.click(
        fn=on_run,
        inputs=[
            segment_picker,
            segments_state,
            ref_frame,
            object_id,
            simplify_factor,
            depth_source,
        ],
        outputs=[status_html, log_box, video_out, mesh_out],
    )

    components.update(
        {
            "segments_path": segments_path,
            "segment_picker": segment_picker,
            "segments_state": segments_state,
            "setup_banner": setup_banner,
            "ref_frame": ref_frame,
            "object_id": object_id,
            "simplify_factor": simplify_factor,
            "depth_source": depth_source,
            "run_btn": run_btn,
            "status_html": status_html,
            "video_out": video_out,
            "mesh_out": mesh_out,
            "log_box": log_box,
            # Exposing on_run lets app.py chain it via .then() so a click
            # on the Retrieve tab's "Reconstruct →" auto-triggers the run.
            "on_run": on_run,
        }
    )
    return components
