# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tab implementations for the webapp."""

from video_ingestion_agent.webapp.tabs.database_tab import create_database_tab
from video_ingestion_agent.webapp.tabs.ingestion_tab import create_ingestion_tab
from video_ingestion_agent.webapp.tabs.query_tab import create_query_tab
from video_ingestion_agent.webapp.tabs.reconstruction_tab import create_reconstruction_tab
from video_ingestion_agent.webapp.tabs.settings_tab import create_settings_tab

__all__ = [
    "create_ingestion_tab",
    "create_query_tab",
    "create_database_tab",
    "create_reconstruction_tab",
    "create_settings_tab",
]
