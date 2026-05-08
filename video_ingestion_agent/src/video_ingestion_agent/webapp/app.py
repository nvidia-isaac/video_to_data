# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main Gradio application for video_ingestion_agent."""

import logging
from pathlib import Path

import gradio as gr

from video_ingestion_agent.webapp.config import AppConfig

logger = logging.getLogger(__name__)


def create_app(config: AppConfig = None) -> gr.Blocks:
    """Create the main Gradio application.

    Args:
        config: Application configuration. Uses defaults if not provided.

    Returns:
        Configured Gradio Blocks application.
    """
    if config is None:
        # Load config with defaults from project config files
        config = AppConfig.from_project_configs()

    # Ensure directories exist
    config.ensure_dirs()

    clips_abs = str(Path(config.default_clips_dir).resolve())
    gr.set_static_paths(paths=[clips_abs])
    logger.info(f"Registered static file path: {clips_abs}")

    # Import tab creators
    from .tabs import (
        create_database_tab,
        create_ingestion_tab,
        create_query_tab,
        create_reconstruction_tab,
    )

    # Services dict (minimal for now)
    services: dict = {
        "config": config,
    }

    # Wire reconstruction service if a `reconstruction:` block was configured.
    # The stage modules live inside this package (reconstruction_interface/),
    # so no cross-package path is needed — `python -m` resolves them.
    if config.reconstruction:
        from .services import ReconstructionConfig, ReconstructionService

        try:
            recon_cfg = ReconstructionConfig.from_dict(config.reconstruction)
            services["reconstruction"] = ReconstructionService(recon_cfg)
            logger.info("Reconstruction service wired")
        except Exception as e:
            logger.warning(f"Reconstruction service unavailable: {e}")

    with gr.Blocks(title="Video Ingestion Agent") as app:
        # Header
        gr.HTML(
            '<div class="app-header">'
            '<span class="nvidia-wordmark">NVIDIA</span>'
            '<span class="app-title">Video Ingestion Agent</span>'
            "</div>"
        )

        # Main tabs — keep a handle on the Tabs container so cross-tab
        # handlers (e.g. Reconstruct → from query results) can switch tabs.
        with gr.Tabs() as tabs:
            with gr.Tab("Retrieve", id="query"):
                query_components = create_query_tab(services, config)

            with gr.Tab("Database", id="database"):
                create_database_tab(services, config)

            with gr.Tab("Ingest", id="ingest"):
                create_ingestion_tab(services, config)

            with gr.Tab("Reconstruct", id="reconstruct"):
                recon_components = create_reconstruction_tab(services, config)

        # ── Cross-tab: Query "Reconstruct →" button feeds into Reconstruct tab ──
        if (
            "reconstruct_btn" in query_components
            and "segments_state" in recon_components
            and "reconstruction" in services
        ):

            def _jump_to_reconstruct(picker_value, query_result):
                """Build a one-segment payload from the picked query clip
                and prefill the Reconstruct tab. Switches tabs in the same
                handler so the user lands on the Reconstruct view ready
                to click Run."""
                if not picker_value or query_result is None:
                    return (
                        gr.update(),  # tabs (no-op)
                        gr.update(),  # segments_state
                        gr.update(),  # segment_picker
                        gr.update(),  # setup_banner
                    )
                try:
                    idx = int(picker_value)
                except (TypeError, ValueError):
                    return (gr.update(), gr.update(), gr.update(), gr.update())

                clips = getattr(query_result, "clips", []) or []
                if not (0 <= idx < len(clips)):
                    return (gr.update(), gr.update(), gr.update(), gr.update())

                clip = dict(clips[idx])
                # Synthesize the keys the Reconstruct tab's segments_state expects
                # (it speaks the clips_final.jsonl shape: clip_id / start_t / end_t).
                from video_ingestion_agent.webapp.models.reconstruction import (
                    ReconstructionRequest,
                )

                req = ReconstructionRequest.from_clip_dict(clip)
                # Overwrite clip_id (not setdefault) with the prefixed
                # segment_id so the dropdown label and the on_run lookup
                # both reference the same string. Raw clip_ids from query
                # results aren't unique across videos.
                clip["clip_id"] = req.segment_id
                clip.setdefault("start_t", req.start_t)
                clip.setdefault("end_t", req.end_t)

                label = f"{req.segment_id}  —  {req.action_label} ({req.object_label})"
                return (
                    gr.update(selected="reconstruct"),  # tabs → Reconstruct
                    [clip],  # segments_state
                    gr.update(choices=[label], value=label),  # segment_picker
                    gr.update(
                        value=f"Loaded segment **{req.segment_id}** from query "
                        f"results — click **Run reconstruction** to start.",
                        visible=True,
                    ),
                )

            query_components["reconstruct_btn"].click(
                fn=_jump_to_reconstruct,
                inputs=[
                    query_components["reconstruct_picker"],
                    query_components["query_result_state"],
                ],
                outputs=[
                    tabs,
                    recon_components["segments_state"],
                    recon_components["segment_picker"],
                    recon_components["setup_banner"],
                ],
            ).then(
                # Auto-trigger the chain so the user doesn't have to click Run
                # a second time. Gradio re-reads inputs at chain time, so this
                # picks up the segment_picker value that _jump_to_reconstruct
                # just set, plus the user's current depth_source.
                fn=recon_components["on_run"],
                inputs=[
                    recon_components["segment_picker"],
                    recon_components["segments_state"],
                    recon_components["ref_frame"],
                    recon_components["object_id"],
                    recon_components["simplify_factor"],
                    recon_components["depth_source"],
                ],
                outputs=[
                    recon_components["status_html"],
                    recon_components["log_box"],
                    recon_components["video_out"],
                    recon_components["mesh_out"],
                ],
            )

    return app


def main():
    """Launch the application."""
    import argparse

    parser = argparse.ArgumentParser(description="Video Ingestion Agent Web App")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = AppConfig.from_file(args.config) if args.config else AppConfig.from_project_configs()

    # Create and launch app
    app = create_app(config)

    logger.info(f"Starting Video Ingestion Agent webapp on {args.host}:{args.port}")

    app.queue()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=config.build_theme(),
        css=config.custom_css,
        head=config.dark_mode_head,
        allowed_paths=[str(Path(config.default_clips_dir).resolve())],
    )


if __name__ == "__main__":
    main()
