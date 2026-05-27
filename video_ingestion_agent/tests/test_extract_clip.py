# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Unit tests for the extract_clip tool's path-resolution fallback.

Specifically covers the LLM-mutation case where the analyzer rewrites
a relative path (``data/foo.mp4``) as absolute (``/data/foo.mp4``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from video_ingestion_agent.retrieval.tools.extract_clip import ExtractClipTool


@pytest.fixture
def fake_video(tmp_path, monkeypatch):
    """Create a non-empty file at a relative path, with cwd set so the
    relative form (``data/foo.mp4``) resolves but the absolute form
    (``/data/foo.mp4``) does not."""
    monkeypatch.chdir(tmp_path)
    rel_dir = Path("data/test_videos")
    rel_dir.mkdir(parents=True)
    video = rel_dir / "foo.mp4"
    video.write_bytes(b"\x00")
    return str(video)  # "data/test_videos/foo.mp4"


def test_get_video_path_returns_existing_relative(fake_video):
    tool = ExtractClipTool(output_dir="outputs/clips_test")
    assert tool._get_video_path(video_path=fake_video) == fake_video


def test_get_video_path_strips_corrupted_leading_slash(fake_video):
    """LLM mutates `data/...` -> `/data/...` — fallback should recover."""
    tool = ExtractClipTool(output_dir="outputs/clips_test")
    corrupted = "/" + fake_video
    assert tool._get_video_path(video_path=corrupted) == fake_video


def test_get_video_path_falls_through_to_registry_when_path_invalid(fake_video):
    """If neither raw nor stripped path exists, video_id registry wins."""
    tool = ExtractClipTool(video_paths={42: fake_video}, output_dir="outputs/clips_test")
    bogus = "/no/such/path/movie.mp4"
    assert tool._get_video_path(video_id=42, video_path=bogus) == fake_video


def test_get_video_path_returns_unmodified_when_recovery_fails():
    """No corruption recovery available — return the input path so the
    downstream existence check raises the canonical error."""
    tool = ExtractClipTool(output_dir="outputs/clips_test")
    bogus = "/totally/missing.mp4"
    # No video_id registry, no default_video_path: returns bogus as-is.
    assert tool._get_video_path(video_path=bogus) == bogus
