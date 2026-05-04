# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`ClipDeduplicator`.

Covers both heuristic (no model) and LLM-based (with model) dedup strategies.
"""

import json
from unittest.mock import MagicMock

import pytest

from video_ingestion_agent.ingestion.segmentation.dedup import ClipDeduplicator
from video_ingestion_agent.ingestion.state import ClipContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(clip_id: str, start: float, end: float, action: str = "act") -> ClipContext:
    """Convenience factory for test clips."""
    return ClipContext(
        clip_id=clip_id,
        video_path="test.mp4",
        start_t=start,
        end_t=end,
        object="obj",
        action=action,
        description="desc",
    )


def _mock_model(responses: list[str]) -> MagicMock:
    """Create a mock BaseModel that returns canned generate_text responses."""
    model = MagicMock()
    model.generate_text = MagicMock(side_effect=responses)
    return model


def _heuristic(threshold: float | None = 0.1) -> ClipDeduplicator:
    return ClipDeduplicator(overlap_threshold=threshold, method="heuristic")


def _llm(threshold: float | None, model: MagicMock) -> ClipDeduplicator:
    return ClipDeduplicator(overlap_threshold=threshold, method="llm", model=model)


# ============================================================================
# Heuristic dedup (no model)
# ============================================================================


class TestHeuristicDedup:
    """Tests for the heuristic overlap-merge (no LLM)."""

    def test_no_overlap_keeps_all(self):
        clips = [_clip("a", 0.0, 3.0), _clip("b", 5.0, 8.0), _clip("c", 10.0, 13.0)]
        assert len(_heuristic().run(clips)) == 3

    def test_identical_clips_merged(self):
        clips = [_clip("a", 0.0, 5.0), _clip("b", 0.0, 5.0)]
        assert len(_heuristic().run(clips)) == 1

    def test_overlap_above_threshold_merges(self):
        clips = [
            _clip("long", 0.0, 10.0, action="long_act"),
            _clip("short", 1.0, 9.0, action="short_act"),
        ]
        result = _heuristic().run(clips)
        assert len(result) == 1
        assert result[0].start_t == 0.0
        assert result[0].end_t == 10.0
        assert result[0].action == "long_act"

    def test_overlap_below_threshold_kept(self):
        clips = [_clip("a", 0.0, 5.0), _clip("b", 4.95, 10.0)]
        assert len(_heuristic().run(clips)) == 2

    def test_disabled_when_threshold_none(self):
        clips = [_clip("a", 0.0, 5.0), _clip("b", 0.0, 5.0)]
        assert len(_heuristic(threshold=None).run(clips)) == 2

    def test_single_clip(self):
        assert len(_heuristic().run([_clip("a", 0.0, 5.0)])) == 1

    def test_empty_list(self):
        assert _heuristic().run([]) == []

    def test_result_sorted_by_start_time(self):
        clips = [_clip("c", 20.0, 25.0), _clip("a", 0.0, 5.0), _clip("b", 10.0, 15.0)]
        starts = [c.start_t for c in _heuristic().run(clips)]
        assert starts == [0.0, 10.0, 20.0]

    def test_chunk_overlap_scenario(self):
        clips = [
            _clip("chunk1_clip", 8.0, 14.0, action="chunk1"),
            _clip("chunk2_clip", 9.0, 14.5, action="chunk2"),
        ]
        result = _heuristic().run(clips)
        assert len(result) == 1
        assert result[0].start_t == 8.0
        assert result[0].end_t == 14.5
        assert result[0].action == "chunk1"

    def test_containment_merges(self):
        clips = [
            _clip("big", 0.0, 10.0, action="big"),
            _clip("small", 3.0, 4.0, action="small"),
        ]
        result = _heuristic().run(clips)
        assert len(result) == 1
        assert result[0].action == "big"

    def test_chain_merge(self):
        clips = [_clip("a", 0.0, 3.0), _clip("b", 2.5, 5.0), _clip("c", 4.5, 7.0)]
        result = _heuristic().run(clips)
        assert len(result) == 1
        assert result[0].start_t == 0.0
        assert result[0].end_t == 7.0

    def test_keeps_longer_clip_annotations(self):
        clips = [
            _clip("short", 0.0, 2.0, action="short_act"),
            _clip("long", 1.5, 8.0, action="long_act"),
        ]
        result = _heuristic().run(clips)
        assert len(result) == 1
        assert result[0].action == "long_act"

    def test_preserves_clip_data(self):
        clip = ClipContext(
            clip_id="test",
            video_path="/path/to/video.mp4",
            start_t=0.0,
            end_t=5.0,
            object="metal pan",
            action="wash",
            description="Person washes the pan",
            metadata={"refined": True, "chunk_idx": 2},
        )
        result = _heuristic().run([clip])
        assert len(result) == 1
        assert result[0].object == "metal pan"
        assert result[0].action == "wash"
        assert result[0].metadata["refined"] is True

    def test_union_extends_boundaries(self):
        clips = [_clip("a", 1.0, 6.0), _clip("b", 5.0, 12.0)]
        result = _heuristic().run(clips)
        assert len(result) == 1
        assert result[0].start_t == pytest.approx(1.0)
        assert result[0].end_t == pytest.approx(12.0)


# ============================================================================
# LLM-based dedup (with model)
# ============================================================================


class TestLLMDedup:
    """Tests for the LLM-based overlap merge."""

    def test_no_overlap_no_llm_call(self):
        model = _mock_model([])
        clips = [_clip("a", 0.0, 3.0), _clip("b", 5.0, 8.0)]
        result = _llm(0.1, model).run(clips)
        assert len(result) == 2
        model.generate_text.assert_not_called()

    def test_llm_says_merge(self):
        resp = json.dumps(
            {
                "merge": True,
                "action": "combined_action",
                "object": "combined_object",
                "description": "combined_description",
            }
        )
        model = _mock_model([resp])
        clips = [
            _clip("a", 0.0, 5.0, action="act_a"),
            _clip("b", 4.0, 8.0, action="act_b"),
        ]
        result = _llm(0.1, model).run(clips)
        assert len(result) == 1
        assert result[0].start_t == 0.0
        assert result[0].end_t == 8.0
        assert result[0].action == "combined_action"
        assert result[0].object == "combined_object"
        assert result[0].description == "combined_description"

    def test_llm_says_no_merge(self):
        resp = json.dumps({"merge": False})
        model = _mock_model([resp])
        clips = [
            _clip("a", 0.0, 5.0, action="act_a"),
            _clip("b", 4.0, 8.0, action="act_b"),
        ]
        result = _llm(0.1, model).run(clips)
        assert len(result) == 2
        assert result[0].action == "act_a"
        assert result[1].action == "act_b"

    def test_llm_failure_keeps_clips_separate(self):
        model = MagicMock()
        model.generate_text = MagicMock(side_effect=RuntimeError("API error"))
        clips = [
            _clip("a", 0.0, 5.0, action="long_act"),
            _clip("b", 4.0, 6.0, action="short_act"),
        ]
        result = _llm(0.1, model).run(clips)
        assert len(result) == 2
        assert result[0].action == "long_act"
        assert result[1].action == "short_act"

    def test_llm_bad_json_keeps_clips_separate(self):
        model = _mock_model(["this is not json"])
        clips = [
            _clip("a", 0.0, 5.0, action="long_act"),
            _clip("b", 4.0, 6.0, action="short_act"),
        ]
        result = _llm(0.1, model).run(clips)
        assert len(result) == 2
        assert result[0].action == "long_act"
        assert result[1].action == "short_act"

    def test_single_clip_no_llm_call(self):
        model = _mock_model([])
        result = _llm(0.1, model).run([_clip("a", 0.0, 5.0)])
        assert len(result) == 1
        model.generate_text.assert_not_called()

    def test_disabled_when_threshold_none(self):
        model = _mock_model([])
        clips = [_clip("a", 0.0, 5.0), _clip("b", 4.0, 8.0)]
        result = _llm(None, model).run(clips)
        assert len(result) == 2
        model.generate_text.assert_not_called()

    def test_chain_merge_multiple_llm_calls(self):
        responses = [
            json.dumps({"merge": True, "action": "ab", "object": "o", "description": "d"}),
            json.dumps({"merge": True, "action": "abc", "object": "o", "description": "d"}),
        ]
        model = _mock_model(responses)
        clips = [_clip("a", 0.0, 3.0), _clip("b", 2.5, 5.0), _clip("c", 4.5, 7.0)]
        result = _llm(0.1, model).run(clips)
        assert len(result) == 1
        assert result[0].start_t == 0.0
        assert result[0].end_t == 7.0

    def test_mixed_merge_and_keep(self):
        responses = [
            json.dumps({"merge": True, "action": "merged", "object": "o", "description": "d"}),
            json.dumps({"merge": False}),
        ]
        model = _mock_model(responses)
        clips = [_clip("a", 0.0, 3.0), _clip("b", 2.5, 5.0), _clip("c", 4.8, 8.0)]
        result = _llm(0.1, model).run(clips)
        assert len(result) == 2
        assert result[0].action == "merged"
        assert result[1].start_t == pytest.approx(4.8)

    def test_sorted_output(self):
        model = _mock_model([])
        clips = [_clip("b", 10.0, 15.0), _clip("a", 0.0, 5.0), _clip("c", 20.0, 25.0)]
        result = _llm(0.1, model).run(clips)
        assert [c.start_t for c in result] == [0.0, 10.0, 20.0]


# ============================================================================
# Method validation
# ============================================================================


class TestMethodValidation:
    """Ensure the method parameter is validated correctly."""

    def test_llm_without_model_raises(self):
        with pytest.raises(ValueError, match="requires a model"):
            ClipDeduplicator(overlap_threshold=0.1, method="llm")

    def test_heuristic_without_model_ok(self):
        d = ClipDeduplicator(overlap_threshold=0.1, method="heuristic")
        assert d.method == "heuristic"
