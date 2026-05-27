# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Video Ingestion Agent Web Application.

A Gradio-based web interface for video ingestion, querying, and clip management.
"""

from video_ingestion_agent.webapp.app import create_app, main
from video_ingestion_agent.webapp.config import AppConfig

__all__ = ["create_app", "main", "AppConfig"]
