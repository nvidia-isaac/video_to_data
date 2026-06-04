# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Tests for the ingestion subpackage.

These tests validate imports, config loading, type conversions,
IO utilities, pipeline graph creation, and component initialization
without requiring GPU models or actual video files.
"""

from pathlib import Path

import pytest

# ========================================================================
# 1. Import tests
# ========================================================================


class TestImports:
    """Verify all ingestion modules import cleanly."""

    def test_import_package(self):
        pass

    def test_import_config(self):
        pass

    def test_import_types(self):
        pass

    def test_import_io(self):
        pass

    def test_import_video_utils(self):
        pass

    def test_import_segmenter(self):
        pass

    def test_import_critic(self):
        pass

    def test_import_strategies(self):
        pass

    def test_import_refiner(self):
        pass

    def test_import_entity_nodes(self):
        pass

    def test_import_pipeline(self):
        pass

    def test_import_report(self):
        pass


class TestSegmentationImports:
    """Verify all segmentation subpackage modules import cleanly."""

    def test_import_segmentation_package(self):
        import video_ingestion_agent.ingestion.segmentation  # noqa: F401

    def test_import_segmenter(self):
        from video_ingestion_agent.ingestion.segmentation.dedup import (
            ClipDeduplicator,  # noqa: F401
        )
        from video_ingestion_agent.ingestion.segmentation.segmenter import (  # noqa: F401
            HybridSegmenter,
            parse_json_response,
            parse_timestamp,
        )

    def test_import_critic(self):
        from video_ingestion_agent.ingestion.segmentation.critic import Critic  # noqa: F401

    def test_import_refiner(self):
        from video_ingestion_agent.ingestion.segmentation.refiner import (  # noqa: F401
            refine_clips,
        )

    def test_import_strategies(self):
        from video_ingestion_agent.ingestion.segmentation.strategies import (  # noqa: F401
            ReannotateStrategy,
        )

    def test_import_video_utils(self):
        from video_ingestion_agent.ingestion.segmentation.video_utils import (  # noqa: F401
            cleanup_temp_clips,
            extract_clip_ffmpeg,
            extract_temp_clips,
        )


class TestEntityGraphImports:
    """Verify all entity_graph subpackage modules import cleanly."""

    def test_old_toplevel_path_raises(self):
        """Old video_ingestion_agent.entity_graph path should not exist."""
        with pytest.raises(ModuleNotFoundError):
            import video_ingestion_agent.entity_graph  # noqa: F401

    def test_import_entity_graph_package(self):
        import video_ingestion_agent.ingestion.entity_graph  # noqa: F401

    def test_import_database_writer(self):
        from video_ingestion_agent.ingestion.entity_graph.database_writer import (  # noqa: F401
            DatabaseWriter,
        )

    def test_import_entity_linker(self):
        from video_ingestion_agent.ingestion.entity_graph.entity_linker import (  # noqa: F401
            EntityLinker,
        )

    def test_import_action_segment(self):
        from video_ingestion_agent.ingestion.state import ActionSegment  # noqa: F401

    def test_import_extractors(self):
        from video_ingestion_agent.ingestion.entity_graph.extractors import (  # noqa: F401
            EntityExtractor,
            VisualExtractor,
        )

    def test_import_prompts(self):
        from video_ingestion_agent.ingestion.entity_graph.prompts import (  # noqa: F401
            DEFAULT_ENTITY_TYPES,
            DEFAULT_RELATIONSHIP_TYPES,
            ENTITY_EXTRACTION_SYSTEM_PROMPT,
            ENTITY_EXTRACTION_USER_PROMPT,
        )


# ========================================================================
# 2. Config tests
# ========================================================================


class TestConfig:
    """Test configuration loading and defaults."""

    def test_default_pipeline_config(self):
        from video_ingestion_agent.ingestion.config import PipelineConfig

        config = PipelineConfig()
        assert config.models.vlm_model == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.models.vlm_backend == "local"
        assert config.segmentation.chunk_size == 15.0
        assert config.segmentation.chunk_overlap == 1.5
        assert config.segmentation.min_clip_s == 1.0
        assert config.segmentation.max_clip_s == 30.0
        assert config.verification.max_iterations == 3
        assert config.enable_verification is True
        assert config.enable_refinement is True
        assert config.enable_entity_graph is True
        assert config.enable_reporting is True

    def test_load_config_from_yaml(self):
        from video_ingestion_agent.ingestion.config import load_config

        config_path = Path(__file__).parent.parent / "configs" / "ingestion.yaml"
        if config_path.exists():
            config = load_config(config_path)
            assert config.models.vlm_model == "Qwen/Qwen3-VL-8B-Instruct"
            assert config.segmentation.chunk_size == 15.0
            assert config.verification.max_iterations == 3
        else:
            pytest.skip("ingestion.yaml not found")

    def test_load_config_missing_file(self):
        from video_ingestion_agent.ingestion.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_config_overrides(self):
        from video_ingestion_agent.ingestion.config import PipelineConfig

        config = PipelineConfig(
            enable_verification=False,
            enable_entity_graph=False,
        )
        assert config.enable_verification is False
        assert config.enable_entity_graph is False
        assert config.enable_refinement is True  # default unchanged

    def test_config_model_dump(self):
        from video_ingestion_agent.ingestion.config import PipelineConfig

        config = PipelineConfig()
        d = config.model_dump()
        assert "models" in d
        assert "segmentation" in d
        assert "verification" in d
        assert d["models"]["vlm_model"] == "Qwen/Qwen3-VL-8B-Instruct"


# ========================================================================
# 3. Type tests
# ========================================================================


class TestTypes:
    """Test ClipContext, VerificationResult, and conversion helpers."""

    def test_clip_context_creation(self):
        from video_ingestion_agent.ingestion.state import ClipContext

        clip = ClipContext(
            clip_id="test_clip_0001",
            video_path="/path/to/video.mp4",
            start_t=5.0,
            end_t=10.0,
            object="red cup",
            action="pick up",
            description="Person picks up a red cup from the table",
        )
        assert clip.clip_id == "test_clip_0001"
        assert clip.start_t == 5.0
        assert clip.end_t == 10.0
        assert clip.duration == 5.0
        assert clip.object == "red cup"
        assert clip.action == "pick up"

    def test_clip_context_defaults(self):
        from video_ingestion_agent.ingestion.state import ClipContext

        clip = ClipContext(
            clip_id="clip_1",
            video_path="video.mp4",
            start_t=0.0,
            end_t=3.0,
        )
        assert clip.object == ""
        assert clip.action == ""
        assert clip.description == ""
        assert clip.metadata == {}

    def test_clip_context_serialization(self):
        from video_ingestion_agent.ingestion.state import ClipContext

        clip = ClipContext(
            clip_id="clip_1",
            video_path="video.mp4",
            start_t=1.5,
            end_t=4.5,
            object="spoon",
            action="stir",
            description="Stirring with a spoon",
            metadata={"fps": 30.0},
        )

        # Serialize and deserialize
        data = clip.model_dump()
        assert data["clip_id"] == "clip_1"
        assert data["start_t"] == 1.5
        assert data["object"] == "spoon"

        clip2 = ClipContext(**data)
        assert clip2.clip_id == clip.clip_id
        assert clip2.duration == clip.duration

    def test_verification_result(self):
        from video_ingestion_agent.ingestion.state import VerificationResult

        result = VerificationResult(
            clip_id="clip_1",
            is_valid=False,
            verification_score=0.4,
            violations=["Incorrect object", "Poor boundary"],
        )
        assert not result.is_valid
        assert len(result.violations) == 2

    def test_clips_to_action_segments(self):
        from video_ingestion_agent.ingestion.state import ClipContext, clips_to_action_segments

        clips = [
            ClipContext(
                clip_id="clip_1",
                video_path="v.mp4",
                start_t=0.0,
                end_t=5.0,
                object="cup",
                action="pick up",
                description="Pick up the cup",
            ),
            ClipContext(
                clip_id="clip_2",
                video_path="v.mp4",
                start_t=5.0,
                end_t=9.0,
                object="plate",
                action="place",
                description="Place on plate",
            ),
        ]

        segments = clips_to_action_segments(clips)
        assert len(segments) == 2
        assert segments[0].segment_id == 1
        assert segments[0].start_t == 0.0
        assert segments[0].end_t == 5.0
        assert segments[0].object_name == "cup"
        assert segments[0].action == "pick up"
        assert segments[1].segment_id == 2
        assert segments[1].start_t == 5.0


# ========================================================================
# 4. IO tests
# ========================================================================


class TestIO:
    """Test JSONL read/write utilities."""

    def test_write_and_read_jsonl(self, tmp_path):
        from video_ingestion_agent.ingestion.io import read_jsonl, write_jsonl

        data = [
            {"id": 1, "name": "clip_1"},
            {"id": 2, "name": "clip_2"},
        ]

        filepath = tmp_path / "test.jsonl"
        write_jsonl(data, filepath)

        read_data = list(read_jsonl(filepath))
        assert len(read_data) == 2
        assert read_data[0]["id"] == 1
        assert read_data[1]["name"] == "clip_2"

    def test_write_and_read_models_jsonl(self, tmp_path):
        from video_ingestion_agent.ingestion.io import write_models_jsonl
        from video_ingestion_agent.ingestion.state import ClipContext

        clips = [
            ClipContext(
                clip_id="clip_1",
                video_path="v.mp4",
                start_t=0.0,
                end_t=5.0,
                object="cup",
                action="pick",
                description="Pick up cup",
            )
        ]

        filepath = tmp_path / "clips.jsonl"
        write_models_jsonl(clips, filepath)

        assert filepath.exists()

        # Read back as raw dict
        from video_ingestion_agent.ingestion.io import read_jsonl

        data = list(read_jsonl(filepath))
        assert len(data) == 1
        assert data[0]["clip_id"] == "clip_1"
        assert data[0]["start_t"] == 0.0

    def test_read_jsonl_missing_file(self, tmp_path):
        from video_ingestion_agent.ingestion.io import read_jsonl

        with pytest.raises(FileNotFoundError):
            list(read_jsonl(tmp_path / "nonexistent.jsonl"))

    def test_write_jsonl_creates_parent_dirs(self, tmp_path):
        from video_ingestion_agent.ingestion.io import write_jsonl

        filepath = tmp_path / "sub" / "dir" / "test.jsonl"
        write_jsonl([{"a": 1}], filepath)
        assert filepath.exists()


# ========================================================================
# 5. Parser tests
# ========================================================================


class TestParsers:
    """Test JSON and timestamp parsing utilities."""

    def test_parse_json_response_code_block(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_json_response

        text = """Here are the segments:
```json
[
  {"clip_id": 1, "start_time": 0, "end_time": 5, "object": "cup", "action": "pick", "description": "pick up cup"}
]
```
"""
        result = parse_json_response(text)
        assert len(result) == 1
        assert result[0]["object"] == "cup"

    def test_parse_json_response_raw(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_json_response

        text = '[{"clip_id": 1, "start_time": 2.0, "end_time": 8.0, "object": "box", "action": "move", "description": "move box"}]'
        result = parse_json_response(text)
        assert len(result) == 1
        assert result[0]["start_time"] == 2.0

    def test_parse_json_response_no_json(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_json_response

        with pytest.raises(ValueError, match="No JSON array found"):
            parse_json_response("No JSON here at all")

    def test_parse_timestamp_seconds(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_timestamp

        assert parse_timestamp(5.0) == 5.0
        assert parse_timestamp(0) == 0.0
        assert parse_timestamp("12.5") == 12.5

    def test_parse_timestamp_mmss(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_timestamp

        assert parse_timestamp("1:30") == 90.0
        assert parse_timestamp("0:05") == 5.0

    def test_parse_timestamp_hhmmss(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_timestamp

        assert parse_timestamp("1:00:00") == 3600.0
        assert parse_timestamp("0:02:30") == 150.0

    def test_parse_critic_response(self):
        from video_ingestion_agent.ingestion.segmentation.critic import parse_critic_response

        text = """```json
{
  "is_correct": true,
  "confidence": 0.9,
  "issues": [],
  "overall_quality": "good"
}
```"""
        result = parse_critic_response(text)
        assert result["is_correct"] is True
        assert result["confidence"] == 0.9


# ========================================================================
# 6. Pipeline graph tests
# ========================================================================


class TestPipelineGraph:
    """Test LangGraph pipeline creation (no execution)."""

    def test_create_full_pipeline(self):
        from video_ingestion_agent.ingestion.ingestion_graph import create_pipeline_graph

        app = create_pipeline_graph(
            enable_verification=True,
            enable_refinement=True,
            enable_entity_graph=True,
            enable_reporting=True,
        )
        assert app is not None

    def test_create_segmentation_only_pipeline(self):
        from video_ingestion_agent.ingestion.ingestion_graph import create_pipeline_graph

        app = create_pipeline_graph(
            enable_verification=False,
            enable_refinement=False,
            enable_entity_graph=False,
            enable_reporting=False,
        )
        assert app is not None

    def test_create_no_refinement_pipeline(self):
        from video_ingestion_agent.ingestion.ingestion_graph import create_pipeline_graph

        app = create_pipeline_graph(
            enable_verification=True,
            enable_refinement=False,
            enable_entity_graph=True,
            enable_reporting=True,
        )
        assert app is not None

    def test_create_no_entity_graph_pipeline(self):
        from video_ingestion_agent.ingestion.ingestion_graph import create_pipeline_graph

        app = create_pipeline_graph(
            enable_verification=True,
            enable_refinement=True,
            enable_entity_graph=False,
            enable_reporting=True,
        )
        assert app is not None

    def test_create_segmentation_report_only(self):
        from video_ingestion_agent.ingestion.ingestion_graph import create_pipeline_graph

        app = create_pipeline_graph(
            enable_verification=False,
            enable_refinement=False,
            enable_entity_graph=False,
            enable_reporting=True,
        )
        assert app is not None


# ========================================================================
# 7. Report tests
# ========================================================================


class TestReport:
    """Test HTML report generation."""

    def test_generate_report_no_verification(self, tmp_path):
        from video_ingestion_agent.ingestion.report import generate_html_report
        from video_ingestion_agent.ingestion.state import ClipContext

        clips = [
            ClipContext(
                clip_id="clip_1",
                video_path="video.mp4",
                start_t=0.0,
                end_t=5.0,
                object="cup",
                action="pick",
                description="Pick up the cup",
            ),
        ]

        report_path = generate_html_report(
            clips=clips,
            run_dir=tmp_path,
            verifications=None,
            config_summary={"chunk_size": 15.0},
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "clip_1" in content
        assert "cup" in content

    def test_generate_report_with_verification(self, tmp_path):
        from video_ingestion_agent.ingestion.report import generate_html_report
        from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult

        clips = [
            ClipContext(
                clip_id="clip_1",
                video_path="video.mp4",
                start_t=0.0,
                end_t=5.0,
                object="cup",
                action="pick",
                description="Pick up",
            ),
            ClipContext(
                clip_id="clip_2",
                video_path="video.mp4",
                start_t=5.0,
                end_t=10.0,
                object="plate",
                action="place",
                description="Place down",
            ),
        ]

        verifications = [
            VerificationResult(
                clip_id="clip_1",
                is_valid=True,
                verification_score=0.95,
            ),
            VerificationResult(
                clip_id="clip_2",
                is_valid=False,
                verification_score=0.3,
                violations=["Wrong object"],
            ),
        ]

        report_path = generate_html_report(
            clips=clips,
            run_dir=tmp_path,
            verifications=verifications,
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "Valid" in content
        assert "Invalid" in content


# ========================================================================
# 8. Video utils tests (no ffmpeg needed)
# ========================================================================


class TestVideoUtils:
    """Test video utility functions (non-ffmpeg parts)."""

    def test_check_ffmpeg_available(self):
        from video_ingestion_agent.ingestion.segmentation.video_utils import check_ffmpeg_available

        # Just verify it runs without error and returns bool
        result = check_ffmpeg_available()
        assert isinstance(result, bool)

    def test_cleanup_nonexistent_dir(self, tmp_path):
        from video_ingestion_agent.ingestion.segmentation.video_utils import cleanup_temp_clips

        nonexistent = tmp_path / "does_not_exist"
        # Should not raise
        cleanup_temp_clips(nonexistent)

    def test_cleanup_existing_dir(self, tmp_path):
        from video_ingestion_agent.ingestion.segmentation.video_utils import cleanup_temp_clips

        temp_dir = tmp_path / "temp_clips"
        temp_dir.mkdir()
        (temp_dir / "dummy.mp4").write_text("fake")
        assert temp_dir.exists()

        cleanup_temp_clips(temp_dir)
        assert not temp_dir.exists()


# ========================================================================
# 9. Refiner tests (unit logic, no model needed)
# ========================================================================


class TestRefiner:
    """Test refiner logic without actual model calls."""

    def test_refine_clips_no_invalids(self):
        from video_ingestion_agent.ingestion.config import PipelineConfig
        from video_ingestion_agent.ingestion.segmentation.refiner import refine_clips
        from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult

        clips = [
            ClipContext(
                clip_id="clip_1",
                video_path="v.mp4",
                start_t=0,
                end_t=5,
                object="cup",
                action="pick",
                description="Pick up cup",
            ),
        ]
        verifications = [
            VerificationResult(
                clip_id="clip_1",
                is_valid=True,
                verification_score=0.9,
            ),
        ]

        config = PipelineConfig()
        updated, refined_ids, responses = refine_clips(
            clips=clips,
            verifications=verifications,
            clip_path_map={},
            config=config,
            iteration=0,
        )

        assert len(updated) == 1
        assert len(refined_ids) == 0
        assert len(responses) == 0

    def test_get_strategy_invalid_name(self):
        from video_ingestion_agent.ingestion.config import PipelineConfig
        from video_ingestion_agent.ingestion.segmentation.refiner import _get_strategy

        config = PipelineConfig()
        with pytest.raises(ValueError, match="Unknown refinement strategy"):
            _get_strategy("nonexistent", config)

    def test_get_strategy_reannotate(self):
        from video_ingestion_agent.ingestion.config import PipelineConfig
        from video_ingestion_agent.ingestion.segmentation.refiner import _get_strategy
        from video_ingestion_agent.ingestion.segmentation.strategies import ReannotateStrategy

        config = PipelineConfig()
        strategy = _get_strategy("reannotate", config)
        assert isinstance(strategy, ReannotateStrategy)
        assert strategy.name == "reannotate"

    def test_get_strategy_boundary_adjust_raises(self):
        from video_ingestion_agent.ingestion.config import PipelineConfig
        from video_ingestion_agent.ingestion.segmentation.refiner import _get_strategy

        config = PipelineConfig()
        with pytest.raises(ValueError, match="Unknown refinement strategy"):
            _get_strategy("boundary_adjust", config)


# ========================================================================
# 10. Verification node regression tests
# ========================================================================


class TestVerificationNode:
    """Regression tests for verification node state merging behavior."""

    def test_filters_stale_previous_verifications(self, tmp_path, monkeypatch):
        from video_ingestion_agent.ingestion.config import PipelineConfig
        from video_ingestion_agent.ingestion.segmentation_nodes import verification_node
        from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult

        # Current clip set only contains clip_a; clip_stale no longer exists.
        clips = [
            ClipContext(
                clip_id="clip_a",
                video_path="v.mp4",
                start_t=0.0,
                end_t=5.0,
                object="cup",
                action="pick",
                description="pick cup",
            )
        ]
        previous_verifications = [
            VerificationResult(
                clip_id="clip_a",
                is_valid=True,
                verification_score=0.9,
            ),
            VerificationResult(
                clip_id="clip_stale",
                is_valid=False,
                verification_score=0.1,
                violations=["stale"],
            ),
        ]

        class FakeCritic:
            def __init__(self, config):
                self.config = config

            def verify_clips_batch(self, clips_to_verify, clip_path_map):
                return [
                    (
                        VerificationResult(
                            clip_id="clip_a",
                            is_valid=True,
                            verification_score=0.95,
                        ),
                        "ok",
                    )
                ]

        import video_ingestion_agent.ingestion.segmentation_nodes as nodes_mod

        monkeypatch.setattr(nodes_mod, "Critic", FakeCritic)

        state = {
            "clips": clips,
            "clip_path_map": {"clip_a": tmp_path / "clip_a.mp4"},
            "config": PipelineConfig(),
            "iteration": 1,
            "refined_clip_ids": ["clip_a"],
            "verifications": previous_verifications,
            "run_dir": tmp_path,
        }
        result = verification_node(state)

        assert {v.clip_id for v in result["verifications"]} == {"clip_a"}
        assert result["refinement_needed"] is False


# ========================================================================
# 9. Refinement strategy registry tests
# ========================================================================


class TestRefinementStrategyRegistry:
    """Verify only reannotate is available after BoundaryAdjustStrategy removal."""

    def test_boundary_adjust_strategy_rejected(self):
        """boundary_adjust should raise ValueError since it was removed."""
        from video_ingestion_agent.ingestion.config import PipelineConfig
        from video_ingestion_agent.ingestion.segmentation.refiner import _get_strategy

        config = PipelineConfig()
        with pytest.raises(ValueError, match="Unknown refinement strategy"):
            _get_strategy("boundary_adjust", config)


# ========================================================================
# 9. Overlap merge dedup tests
# ========================================================================


class TestDedupClips:
    """Tests for ClipDeduplicator (heuristic mode) via test_ingestion."""

    def _make_clip(self, clip_id, start_t, end_t, action="action"):
        from video_ingestion_agent.ingestion.state import ClipContext

        return ClipContext(
            clip_id=clip_id,
            video_path="video.mp4",
            start_t=start_t,
            end_t=end_t,
            action=action,
            object="obj",
            description="desc",
        )

    def _dedup(self, clips, threshold=0.1):
        from video_ingestion_agent.ingestion.segmentation.dedup import ClipDeduplicator

        return ClipDeduplicator(overlap_threshold=threshold, method="heuristic").run(clips)

    def test_no_overlap(self):
        """Non-overlapping clips should not be merged."""
        clips = [self._make_clip("a", 0.0, 5.0), self._make_clip("b", 6.0, 10.0)]
        assert len(self._dedup(clips)) == 2

    def test_small_overlap_below_threshold(self):
        """Overlap below threshold should not merge."""
        clips = [self._make_clip("a", 0.0, 5.0), self._make_clip("b", 4.95, 10.0)]
        assert len(self._dedup(clips)) == 2

    def test_overlap_above_threshold_merges(self):
        """Overlap above threshold should merge into one clip."""
        clips = [
            self._make_clip("a", 0.0, 5.0, action="long_action"),
            self._make_clip("b", 4.5, 8.0, action="short_action"),
        ]
        result = self._dedup(clips)
        assert len(result) == 1
        assert result[0].start_t == 0.0
        assert result[0].end_t == 8.0
        assert result[0].action == "long_action"

    def test_containment_merges(self):
        """A small clip fully inside a large one should be merged."""
        clips = [
            self._make_clip("big", 0.0, 10.0, action="big_action"),
            self._make_clip("small", 3.0, 4.0, action="small_action"),
        ]
        result = self._dedup(clips)
        assert len(result) == 1
        assert result[0].action == "big_action"

    def test_chain_merge(self):
        """A chain of overlapping clips should all merge together."""
        clips = [
            self._make_clip("a", 0.0, 3.0),
            self._make_clip("b", 2.5, 5.0),
            self._make_clip("c", 4.5, 7.0),
        ]
        result = self._dedup(clips)
        assert len(result) == 1
        assert result[0].start_t == 0.0
        assert result[0].end_t == 7.0

    def test_keeps_longer_annotations(self):
        """Merged clip should keep annotations from the longer clip."""
        clips = [
            self._make_clip("short", 0.0, 2.0, action="short"),
            self._make_clip("long", 1.5, 8.0, action="long"),
        ]
        result = self._dedup(clips)
        assert len(result) == 1
        assert result[0].action == "long"
        assert result[0].start_t == 0.0
        assert result[0].end_t == 8.0

    def test_disabled_when_threshold_none(self):
        """Setting threshold to None should disable merging."""
        clips = [self._make_clip("a", 0.0, 5.0), self._make_clip("b", 4.0, 8.0)]
        assert len(self._dedup(clips, threshold=None)) == 2

    def test_single_clip(self):
        """A single clip should be returned as-is."""
        assert len(self._dedup([self._make_clip("a", 0.0, 5.0)])) == 1

    def test_sorted_output(self):
        """Output should be sorted by start_t regardless of input order."""
        clips = [
            self._make_clip("b", 10.0, 15.0),
            self._make_clip("a", 0.0, 5.0),
            self._make_clip("c", 20.0, 25.0),
        ]
        result = self._dedup(clips)
        assert len(result) == 3
        assert result[0].start_t == 0.0
        assert result[1].start_t == 10.0
        assert result[2].start_t == 20.0

    def test_config_default(self):
        """Default dedup_overlap_threshold should be -0.1 and dedup_method 'llm'."""
        from video_ingestion_agent.ingestion.config import PipelineConfig

        config = PipelineConfig()
        assert config.segmentation.dedup_overlap_threshold == -0.1
        assert config.segmentation.dedup_method == "llm"

    def test_llm_missing_merge_key_defaults_to_no_merge(self):
        """LLM dedup should conservatively keep clips separate when 'merge' key is missing."""
        from video_ingestion_agent.ingestion.segmentation.dedup import ClipDeduplicator

        class FakeModel:
            def generate_text(self, conversation, max_new_tokens=512, temperature=0.0):
                return '{"object":"obj","action":"act","description":"desc"}'

        clips = [
            self._make_clip("a", 0.0, 5.0),
            self._make_clip("b", 4.5, 8.0),
        ]
        result = ClipDeduplicator(
            overlap_threshold=0.1,
            method="llm",
            model=FakeModel(),
        ).run(clips)
        assert len(result) == 2


class TestValidClipsFilter:
    """Tests for _valid_clips helper in entity_graph_nodes."""

    def _make_clip(self, clip_id, start, end):
        from video_ingestion_agent.ingestion.state import ClipContext

        return ClipContext(
            clip_id=clip_id,
            video_path="video.mp4",
            start_t=start,
            end_t=end,
            object="obj",
            action="act",
            description="desc",
        )

    def _make_verification(self, clip_id, is_valid):
        from video_ingestion_agent.ingestion.state import VerificationResult

        return VerificationResult(
            clip_id=clip_id,
            is_valid=is_valid,
            verification_score=0.9,
        )

    def test_no_verifications_returns_all_clips(self):
        from video_ingestion_agent.ingestion.entity_graph_nodes import _valid_clips

        clips = [self._make_clip("a", 0, 5), self._make_clip("b", 5, 10)]
        state = {"clips": clips, "verifications": []}
        assert _valid_clips(state) == clips

    def test_filters_invalid_clips(self):
        from video_ingestion_agent.ingestion.entity_graph_nodes import _valid_clips

        clips = [
            self._make_clip("a", 0, 5),
            self._make_clip("b", 5, 10),
            self._make_clip("c", 10, 15),
        ]
        verifications = [
            self._make_verification("a", True),
            self._make_verification("b", False),
            self._make_verification("c", True),
        ]
        state = {"clips": clips, "verifications": verifications}
        result = _valid_clips(state)
        assert len(result) == 2
        assert {c.clip_id for c in result} == {"a", "c"}

    def test_all_invalid_returns_empty(self):
        from video_ingestion_agent.ingestion.entity_graph_nodes import _valid_clips

        clips = [self._make_clip("a", 0, 5)]
        verifications = [self._make_verification("a", False)]
        state = {"clips": clips, "verifications": verifications}
        assert _valid_clips(state) == []


# ========================================================================
# 15. Segmenter failure handling (no fabricated placeholder segments)
# ========================================================================


class TestSegmenterFailureHandling:
    """A failed chunk must drop its contribution, not fabricate a segment.

    Regression: a backend error (e.g. a 401 after retries are exhausted) used
    to return a synthetic "manipulation" clip spanning the whole chunk. That
    placeholder polluted the shared databases with fake action_segments and
    frame embeddings that downstream retrieval treated as real, while the run
    still exited 0 and rendered a normal-looking success report.
    """

    def test_segment_chunk_failure_returns_empty(self):
        from unittest.mock import MagicMock

        from video_ingestion_agent.ingestion.config import PipelineConfig
        from video_ingestion_agent.ingestion.segmentation.segmenter import HybridSegmenter

        config = PipelineConfig()
        segmenter = HybridSegmenter(config)

        # Inject a model that fails the way the API backend surfaces a 401
        # once its internal retry/backoff loop is exhausted. Seeding the
        # private cache makes _get_model() return it without touching the
        # ModelManager (no GPU / network).
        failing_model = MagicMock()
        failing_model.generate_from_video.side_effect = RuntimeError(
            "API request failed after 5 attempts: 401 Client Error: Unauthorized"
        )
        segmenter._model = failing_model

        # chunk_end == video_duration -> no temp-chunk extraction, so the path
        # reaches the VLM call directly without needing a real file or ffmpeg.
        clips = segmenter._segment_chunk(
            video_path="/nonexistent/video.mp4",
            chunk_start=0.0,
            chunk_end=10.0,
            video_duration=10.0,
            fps=30.0,
            global_clip_idx=0,
        )

        assert clips == []
        failing_model.generate_from_video.assert_called_once()
