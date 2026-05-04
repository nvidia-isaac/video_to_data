# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video ingestion tab for uploading and processing videos."""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr

from video_ingestion_agent.webapp.config import AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch ingestion helpers
# ---------------------------------------------------------------------------

_BATCH_SCRIPT: Path | None = None


def _find_batch_script() -> Path | None:
    """Locate ``scripts/run_batch_ingestion.py`` relative to the package."""
    global _BATCH_SCRIPT  # noqa: PLW0603
    if _BATCH_SCRIPT is not None:
        return _BATCH_SCRIPT
    # This file lives at src/video_ingestion_agent/webapp/tabs/ingestion_tab.py
    # Project root is five levels up.
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    candidate = pkg_root / "scripts" / "run_batch_ingestion.py"
    if candidate.exists():
        _BATCH_SCRIPT = candidate
        return _BATCH_SCRIPT
    return None


def create_ingestion_tab(services: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Create the video ingestion tab.

    Args:
        services: Dict of service instances.
        config: Application configuration.

    Returns:
        Dict of component references for external use.
    """
    components = {}

    with gr.Column():
        gr.Markdown("## Video Ingestion")
        gr.Markdown(
            "Process videos to build an entity graph database (`graph.db` + `vector.db`).\n\n"
            "**Two modes:**\n"
            "- **Single file** -- upload a video on the left.\n"
            "- **Batch directory** -- enter a directory path containing videos; "
            "set **Parallel workers** > 1 for multi-GPU processing.\n\n"
            "Choose a configuration YAML, set the output directory, "
            "then click **Start Ingestion**."
        )

        with gr.Row():
            # Left column
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### Input Source")
                    video_upload = gr.File(
                        label="Single file video",
                        file_types=["video"],
                        type="filepath",
                    )
                    videos_dir_input = gr.Textbox(
                        label="Batch directory path",
                        value=config.default_videos_dir,
                        placeholder="/mnt/amlfs/home/.../videos",
                        interactive=True,
                    )

                with gr.Group():
                    gr.Markdown("#### Configuration")
                    config_dropdown = gr.Dropdown(
                        label="Configuration YAML",
                        choices=config.get_config_files(),
                        value=config.default_ingestion_config
                        if Path(config.default_ingestion_config).exists()
                        else None,
                        interactive=True,
                    )

            # Right column
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### Output Settings")
                    output_dir_input = gr.Textbox(
                        label="Output Directory",
                        value=config.default_output_dir,
                        placeholder="outputs/",
                        interactive=True,
                    )
                    append_checkbox = gr.Checkbox(
                        label="Append to existing database",
                        value=False,
                    )
                    num_shards_input = gr.Number(
                        label="Parallel workers (batch)",
                        value=1,
                        minimum=1,
                        maximum=64,
                        precision=0,
                        info="Set >1 for multi-GPU batch ingestion.",
                        interactive=True,
                    )
                    resume_checkbox = gr.Checkbox(
                        label="Resume (skip processed)",
                        value=True,
                    )

                start_btn = gr.Button("Start Ingestion", variant="primary")

        # Log output
        with gr.Group():
            gr.Markdown("#### Log Output")
            progress_display = gr.Markdown(
                value="Ready to ingest video.",
            )
            log_output = gr.Textbox(
                label="Ingestion Logs",
                lines=10,
                interactive=False,
                elem_classes=["log-output"],
            )

        # Per-worker log viewer (for batch ingestion)
        with gr.Accordion("Worker Logs", open=False):
            gr.Markdown(
                "View detailed logs from individual batch ingestion workers. "
                "Select a worker and click **Refresh** (works during and after ingestion)."
            )
            with gr.Row():
                worker_log_dropdown = gr.Dropdown(
                    label="Worker",
                    choices=[],
                    interactive=True,
                    scale=1,
                )
                worker_log_refresh_btn = gr.Button(
                    "Refresh",
                    scale=0,
                    min_width=100,
                )
            worker_log_output = gr.Textbox(
                label="Worker Log",
                lines=20,
                interactive=False,
                max_lines=40,
                elem_classes=["log-output"],
            )

        # Results section (hidden initially)
        with gr.Column(visible=False) as results_section:
            gr.Markdown("### Ingestion Results")
            results_display = gr.JSON(label="Statistics")

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    def _discover_videos_in_dir(dir_path: str) -> list[Path]:
        """Recursively discover video files under *dir_path*."""
        root = Path(dir_path)
        return sorted(
            p.resolve()
            for p in root.rglob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
        )

    # ------------------------------------------------------------------
    # Per-worker log viewer
    # ------------------------------------------------------------------
    def _discover_worker_logs(out_dir: str) -> list[str]:
        """Return sorted list of available worker log labels."""
        out = Path(out_dir)
        if not out.is_dir():
            return []
        labels: list[str] = []
        for lf in sorted(out.glob("worker_*.log")):
            try:
                wid = int(lf.stem.split("_")[-1])
            except ValueError:
                continue
            size_kb = lf.stat().st_size / 1024
            labels.append(f"Worker {wid}  ({size_kb:.0f} KB)")
        return labels

    def _read_worker_log_file(out_dir: str, label: str, tail_lines: int = 500) -> str:
        """Read the log file for the selected worker (last *tail_lines* lines)."""
        if not label:
            return ""
        try:
            wid = int(label.split()[1])
        except (IndexError, ValueError):
            return f"Could not parse worker ID from: {label}"
        log_path = Path(out_dir) / f"worker_{wid}.log"
        if not log_path.exists():
            return f"Log file not found: {log_path}"
        try:
            with open(log_path) as f:
                lines = f.readlines()
            return "".join(lines[-tail_lines:])
        except Exception as e:
            return f"Error reading log: {e}"

    def refresh_worker_logs(out_dir: str, selected_worker: str):
        """Refresh the worker log dropdown choices and load selected log."""
        choices = _discover_worker_logs(out_dir)
        log_text = _read_worker_log_file(out_dir, selected_worker)
        # Keep current selection if still valid, otherwise pick first
        if selected_worker not in choices:
            selected_worker = choices[0] if choices else None
            log_text = _read_worker_log_file(out_dir, selected_worker or "")
        return (
            gr.update(choices=choices, value=selected_worker),
            log_text,
        )

    def on_worker_dropdown_change(out_dir: str, selected_worker: str):
        """Load log when the dropdown selection changes."""
        return _read_worker_log_file(out_dir, selected_worker)

    # ------------------------------------------------------------------
    # Batch ingestion via subprocess
    # ------------------------------------------------------------------
    def _run_batch_subprocess(
        batch_script: Path,
        videos_dir: str,
        config_file: str | None,
        output_dir: str,
        num_shards: int,
        resume: bool,
        progress,
    ):
        """Launch ``run_batch_ingestion.py`` and stream progress back."""
        cfg = config_file or config.default_ingestion_config
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(batch_script),
            "--input-dir",
            videos_dir,
            "-c",
            cfg,
            "--output-dir",
            str(output_path),
            "--num-shards",
            str(num_shards),
        ]
        if resume:
            cmd.append("--resume")

        logs: list[str] = [
            "Batch ingestion (subprocess)",
            f"  Directory : {videos_dir}",
            f"  Workers   : {num_shards}",
            f"  Config    : {cfg}",
            f"  Output    : {output_path}",
            f"  Resume    : {resume}",
            "",
        ]

        yield (
            f"**Launching batch ingestion** with {num_shards} worker(s)...",
            "\n".join(logs),
            gr.update(visible=False),
            None,
        )

        # Start subprocess
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Poll subprocess stdout and worker progress JSONL files.
        # Detailed per-worker logs are viewed separately via the Worker Logs
        # accordion (dropdown + refresh).
        progress_lines_read: dict[int, int] = {}  # worker_id -> jsonl lines consumed
        t_start = time.time()

        def _read_worker_progress() -> tuple[int, int, int]:
            """Read all progress_worker_*.jsonl files and aggregate counts."""
            success = 0
            failed = 0
            total_clips = 0
            for pf in output_path.glob("progress_worker_*.jsonl"):
                try:
                    wid = int(pf.stem.split("_")[-1])
                except ValueError:
                    continue
                start_line = progress_lines_read.get(wid, 0)
                try:
                    with open(pf) as f:
                        all_lines = f.readlines()
                except Exception:
                    continue
                for jline in all_lines[start_line:]:
                    jline = jline.strip()
                    if not jline:
                        continue
                    try:
                        rec = json.loads(jline)
                    except json.JSONDecodeError:
                        continue
                    vid_name = Path(rec.get("video", "")).name
                    status = rec.get("status", "unknown")
                    elapsed_s = rec.get("elapsed_s", 0)
                    clips = rec.get("n_clips", 0)
                    if status == "success":
                        success += 1
                        total_clips += clips
                        logs.append(
                            f"  [Worker {wid}] {vid_name}: {clips} clips ({elapsed_s:.1f}s)"
                        )
                    else:
                        failed += 1
                        err = rec.get("error", "")
                        logs.append(f"  [Worker {wid}] {vid_name}: FAILED - {err}")
                progress_lines_read[wid] = len(all_lines)
            return success, failed, total_clips

        # Stream subprocess output + progress summaries
        done_videos = 0
        failed_videos = 0
        total_clips = 0
        last_poll = 0.0

        while True:
            # Read a line from main subprocess stdout
            retcode = proc.poll()
            line = ""
            if proc.stdout:
                line = proc.stdout.readline()
            if line:
                logs.append(line.rstrip())

            # Periodically poll progress files (every 2s)
            now = time.time()
            if now - last_poll > 2.0:
                last_poll = now

                # Read structured progress
                done_videos, failed_videos, total_clips = _read_worker_progress()

                elapsed = now - t_start
                desc = (
                    f"Batch ingestion: {done_videos} done, {failed_videos} failed ({elapsed:.0f}s)"
                )
                progress(0.5, desc=desc)  # indeterminate-ish

                yield (
                    f"**Batch ingestion running** ({num_shards} workers)\n\n"
                    f"- Videos done: {done_videos}\n"
                    f"- Failed: {failed_videos}\n"
                    f"- Total clips: {total_clips}\n"
                    f"- Elapsed: {elapsed:.0f}s\n\n"
                    f"*Open the **Worker Logs** section below to view "
                    f"detailed per-worker output.*",
                    "\n".join(logs[-300:]),
                    gr.update(visible=False),
                    None,
                )

            if retcode is not None and not line:
                break

            if not line:
                time.sleep(0.5)

        # Final read of progress files
        done_videos, failed_videos, total_clips = _read_worker_progress()
        elapsed = time.time() - t_start

        progress(1.0, desc="Batch ingestion complete")

        graph_db = str(output_path / "graph.db")
        vector_db = str(output_path / "vector.db")

        if retcode != 0:
            logs.append(f"\nBatch process exited with code {retcode}")

        total_processed = done_videos + failed_videos
        summary = (
            f"**Batch Ingestion Complete!**\n\n"
            f"- **Videos processed:** {done_videos}/{total_processed}\n"
            f"- **Failed:** {failed_videos}\n"
            f"- **Total clips:** {total_clips}\n"
            f"- **Workers:** {num_shards}\n"
            f"- **Total time:** {elapsed:.1f}s\n"
            f"- **Exit code:** {retcode}"
        )

        yield (
            summary,
            "\n".join(logs[-500:]),
            gr.update(visible=True),
            {
                "success": failed_videos == 0 and retcode == 0,
                "graph_db": graph_db,
                "vector_db": vector_db,
                "videos_processed": done_videos,
                "videos_failed": failed_videos,
                "total_clips": total_clips,
                "elapsed": round(elapsed, 1),
                "exit_code": retcode,
            },
        )

    # ------------------------------------------------------------------
    # Single-file in-process ingestion
    # ------------------------------------------------------------------
    def _run_single_video(
        video_path: str,
        config_file: str | None,
        output_dir: str,
        progress,
    ):
        """Ingest a single video through the in-process pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_db = str(output_path / "graph.db")
        vector_db = str(output_path / "vector.db")

        video_name = Path(video_path).name
        logs: list[str] = [
            f"Single video ingestion: {video_name}",
            f"Output Dir: {output_dir}",
            f"Graph DB: {output_db}",
            f"Vector DB: {vector_db}",
            "",
        ]

        yield (
            f"**Processing:** `{video_name}`",
            "\n".join(logs),
            gr.update(visible=False),
            None,
        )

        try:
            from ..services import IngestionService

            service = IngestionService(config_file or config.default_ingestion_config)

            def progress_callback(prog):
                pct = int(prog.overall_progress * 100)
                progress(prog.overall_progress, desc=f"{prog.step}: {pct}%")
                logs.append(f"  [{prog.step}] {prog.message}")

            result = service.ingest_video(
                video_path=video_path,
                output_db=output_db,
                vector_db_path=vector_db,
                config_path=config_file,
                progress_callback=progress_callback,
            )

            progress(1.0, desc="Complete")

            if result.success:
                summary = (
                    f"**Ingestion Complete!**\n\n"
                    f"- **Segments:** {result.segment_count}\n"
                    f"- **Entities:** {result.entity_count}\n"
                    f"- **Relationships:** {result.relationship_count}\n"
                    f"- **Video duration:** {result.video_duration:.1f}s\n"
                    f"- **Processing time:** {result.elapsed_time:.1f}s"
                )
                logs.append(
                    f"Done: {result.segment_count} segments, "
                    f"{result.entity_count} entities ({result.elapsed_time:.1f}s)"
                )
                yield (
                    summary,
                    "\n".join(logs),
                    gr.update(visible=True),
                    {
                        "success": True,
                        "graph_db": output_db,
                        "vector_db": vector_db,
                        "segments": result.segment_count,
                        "entities": result.entity_count,
                        "relationships": result.relationship_count,
                        "duration": result.video_duration,
                        "elapsed": result.elapsed_time,
                    },
                )
            else:
                logs.append(f"FAILED: {result.error_message}")
                yield (
                    f"**Error:** {result.error_message}",
                    "\n".join(logs),
                    gr.update(visible=False),
                    None,
                )

        except Exception as e:
            logger.error(f"Ingestion error: {e}", exc_info=True)
            logs.append(f"Error: {str(e)}")
            yield (
                f"**Error:** {str(e)}",
                "\n".join(logs),
                gr.update(visible=False),
                None,
            )

    # ------------------------------------------------------------------
    # Main event handler
    # ------------------------------------------------------------------
    def run_ingestion(
        video_file,
        videos_dir,
        config_file,
        output_dir,
        append_mode,
        num_shards,
        resume,
        progress=gr.Progress(),
    ):
        """Run video ingestion with progress updates.

        Supports two modes:
        1. **Single file** upload via *video_file* -- runs in-process.
        2. **Batch directory** via *videos_dir* -- launches the batch
           ingestion script as a subprocess with parallel workers.
        """
        num_shards = int(num_shards or 1)

        # --- Directory mode -> batch subprocess ---
        if videos_dir and videos_dir.strip():
            vdir = Path(videos_dir.strip())
            if not vdir.exists():
                yield (
                    f"Video directory not found: `{videos_dir}`",
                    "",
                    gr.update(visible=False),
                    None,
                )
                return

            batch_script = _find_batch_script()
            if batch_script is None:
                yield (
                    "**Error:** Could not locate `scripts/run_batch_ingestion.py`. "
                    "Make sure the script exists in the project root.",
                    "",
                    gr.update(visible=False),
                    None,
                )
                return

            yield from _run_batch_subprocess(
                batch_script=batch_script,
                videos_dir=videos_dir.strip(),
                config_file=config_file,
                output_dir=output_dir,
                num_shards=num_shards,
                resume=resume,
                progress=progress,
            )
            return

        # --- Single file mode -> in-process ---
        if video_file:
            yield from _run_single_video(
                video_path=video_file,
                config_file=config_file,
                output_dir=output_dir,
                progress=progress,
            )
            return

        yield (
            "Please upload a video or provide a video directory path.",
            "",
            gr.update(visible=False),
            None,
        )

    # Wire up events
    start_btn.click(
        fn=run_ingestion,
        inputs=[
            video_upload,
            videos_dir_input,
            config_dropdown,
            output_dir_input,
            append_checkbox,
            num_shards_input,
            resume_checkbox,
        ],
        outputs=[progress_display, log_output, results_section, results_display],
    )

    # Worker log viewer events
    worker_log_refresh_btn.click(
        fn=refresh_worker_logs,
        inputs=[output_dir_input, worker_log_dropdown],
        outputs=[worker_log_dropdown, worker_log_output],
    )

    worker_log_dropdown.change(
        fn=on_worker_dropdown_change,
        inputs=[output_dir_input, worker_log_dropdown],
        outputs=[worker_log_output],
    )

    components["video_upload"] = video_upload
    components["videos_dir_input"] = videos_dir_input
    components["start_btn"] = start_btn
    components["output_dir_input"] = output_dir_input
    components["num_shards_input"] = num_shards_input
    components["resume_checkbox"] = resume_checkbox

    return components
