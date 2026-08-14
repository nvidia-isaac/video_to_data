# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Tests for visual-search segment resolution in ExecutorNode."""

from dataclasses import dataclass

from video_ingestion_agent.retrieval.config import RetrievalConfig
from video_ingestion_agent.retrieval.nodes.executor import ExecutorNode
from video_ingestion_agent.retrieval.tools.base import ToolResult
from video_ingestion_agent.retrieval.tools.search_graph import SegmentResult


@dataclass
class _FakeFrame:
    frame_id: str
    video_id: str
    video_path: str | None
    timestamp: float
    similarity: float
    segment_id: str | None


class _FakeSearchFramesTool:
    def __init__(self, frames):
        self.frames = frames

    def execute(self, **kwargs: object):
        return ToolResult(success=True, data=self.frames)


class _FakeSearchGraphTool:
    def __init__(self, segments):
        self.segments = segments
        self.last_kwargs = {}

    def get_segments_overlapping(self, **kwargs: object):
        self.last_kwargs = kwargs
        return self.segments


def test_visual_search_passes_video_filters_to_overlap_query():
    frames = [
        _FakeFrame(
            frame_id="f1",
            video_id="video_stem",
            video_path="/tmp/video_a.mp4",
            timestamp=12.0,
            similarity=0.95,
            segment_id="clip-1",
        )
    ]
    graph_tool = _FakeSearchGraphTool(
        [
            SegmentResult(
                segment_id=1,
                action="pick_up",
                object_name="mug",
                start_t=10.0,
                end_t=15.0,
                description="pick up mug",
                video_id=3,
                video_path="/tmp/video_a.mp4",
            )
        ]
    )
    node = ExecutorNode(
        config=RetrievalConfig(),
        tools={
            "search_frames": _FakeSearchFramesTool(frames),
            "search_graph": graph_tool,
        },
    )

    text = node._search_visual({"search_type": "visual", "object_name": "mug"}, "mug")
    assert "Segment clip-1 [10.0s - 15.0s]" in text
    assert graph_tool.last_kwargs["video_path"] == "/tmp/video_a.mp4"
    assert graph_tool.last_kwargs["video_id"] is None


def test_visual_search_selects_best_overlap_segment_not_first():
    frames = [
        _FakeFrame(
            frame_id="f1",
            video_id="3",
            video_path="/tmp/video_b.mp4",
            timestamp=12.0,
            similarity=0.91,
            segment_id="clip-2",
        ),
        _FakeFrame(
            frame_id="f2",
            video_id="3",
            video_path="/tmp/video_b.mp4",
            timestamp=13.0,
            similarity=0.89,
            segment_id="clip-2",
        ),
    ]

    # First segment overlaps less; second overlaps more and should be selected.
    graph_tool = _FakeSearchGraphTool(
        [
            SegmentResult(
                segment_id=101,
                action="place",
                object_name="mug",
                start_t=8.0,
                end_t=12.1,
                description="low-overlap candidate",
                video_id=3,
                video_path="/tmp/video_b.mp4",
            ),
            SegmentResult(
                segment_id=102,
                action="place",
                object_name="mug",
                start_t=11.8,
                end_t=14.0,
                description="best-overlap candidate",
                video_id=3,
                video_path="/tmp/video_b.mp4",
            ),
        ]
    )
    node = ExecutorNode(
        config=RetrievalConfig(),
        tools={
            "search_frames": _FakeSearchFramesTool(frames),
            "search_graph": graph_tool,
        },
    )

    text = node._search_visual({"search_type": "visual", "object_name": "mug"}, "mug")
    assert "best-overlap candidate" in text
    assert "low-overlap candidate" not in text
    assert graph_tool.last_kwargs["video_id"] == 3
