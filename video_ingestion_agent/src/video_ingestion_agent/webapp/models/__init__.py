# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for the webapp."""

from video_ingestion_agent.webapp.models.query_history import ClipResult, QueryRecord, SubTaskResult

__all__ = [
    "QueryRecord",
    "ClipResult",
    "SubTaskResult",
]
