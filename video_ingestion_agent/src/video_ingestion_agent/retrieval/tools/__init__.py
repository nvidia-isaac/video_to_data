# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tools for agentic video retrieval workflows.

Available tools:
- SearchGraphTool: Query entity graph (entities, relationships, segments)
- SearchFramesTool: Semantic search over frame embeddings
- ExtractClipTool: Extract video clips by timestamp
"""

from video_ingestion_agent.retrieval.tools.base import BaseTool, ToolResult
from video_ingestion_agent.retrieval.tools.extract_clip import ClipInfo, ExtractClipTool
from video_ingestion_agent.retrieval.tools.search_frames import FrameSearchResult, SearchFramesTool
from video_ingestion_agent.retrieval.tools.search_graph import (
    EntityResult,
    RelationshipResult,
    SearchGraphTool,
    SegmentResult,
)

__all__ = [
    # Base
    "BaseTool",
    "ToolResult",
    # Search Graph
    "SearchGraphTool",
    "EntityResult",
    "RelationshipResult",
    "SegmentResult",
    # Search Frames
    "SearchFramesTool",
    "FrameSearchResult",
    # Extract Clip
    "ExtractClipTool",
    "ClipInfo",
]
