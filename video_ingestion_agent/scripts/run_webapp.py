#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the Video Ingestion Agent web application.

This script is a launcher for the Gradio webapp defined in
video_ingestion_agent.webapp.app.

Usage:
    python scripts/run_webapp.py [options]

Options:
    --port PORT     Port to run on (default: 7860)
    --host HOST     Host to bind to (default: 127.0.0.1)
    --share         Create public link via Gradio
    --config FILE   Path to config file
    --debug         Enable debug logging
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from video_ingestion_agent.webapp.app import create_app  # noqa: E402
from video_ingestion_agent.webapp.config import AppConfig  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    """Launch the Video Ingestion Agent webapp."""
    parser = argparse.ArgumentParser(
        description="Video Ingestion Agent Web App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument("--share", action="store_true", help="Create public link via Gradio")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    if args.config:
        config = AppConfig.from_file(args.config)
    else:
        config = AppConfig.from_project_configs()

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
