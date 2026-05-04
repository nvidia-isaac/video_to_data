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
    )

    # Services dict (minimal for now)
    services = {
        "config": config,
    }

    with gr.Blocks(title="Video Ingestion Agent") as app:
        # Header
        gr.HTML(
            '<div class="app-header">'
            '<span class="nvidia-wordmark">NVIDIA</span>'
            '<span class="app-title">Video Ingestion Agent</span>'
            "</div>"
        )

        # Main tabs
        with gr.Tabs():
            with gr.Tab("Retrieve", id="query"):
                create_query_tab(services, config)

            with gr.Tab("Database", id="database"):
                create_database_tab(services, config)

            with gr.Tab("Ingest", id="ingest"):
                create_ingestion_tab(services, config)

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
