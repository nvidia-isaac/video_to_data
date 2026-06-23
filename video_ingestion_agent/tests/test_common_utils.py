# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Tests for the refactored shared utilities in common/.

Covers:
- common.parsing: parse_llm_json, parse_timestamp
- common.video_utils: get_video_info, extract_frames_base64, extract_clip_ffmpeg
- Backward compatibility of old call sites (segmenter, critic, strategies, etc.)
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

# ========================================================================
# 1. common.parsing.parse_llm_json
# ========================================================================


class TestParseLlmJson:
    """Tests for the shared parse_llm_json utility."""

    def test_json_array_in_code_block(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        text = '```json\n[{"a": 1}, {"b": 2}]\n```'
        result = parse_llm_json(text, expect_array=True)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_json_object_in_code_block(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        text = '```json\n{"key": "value", "num": 42}\n```'
        result = parse_llm_json(text, expect_array=False)
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_raw_json_array(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        text = 'Here are results:\n[{"x": 1}]'
        result = parse_llm_json(text, expect_array=True)
        assert isinstance(result, list)
        assert result[0]["x"] == 1

    def test_raw_json_object(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        text = 'The analysis is: {"is_correct": true, "confidence": 0.9}'
        result = parse_llm_json(text, expect_array=False)
        assert isinstance(result, dict)
        assert result["is_correct"] is True

    def test_no_json_raises_valueerror(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        with pytest.raises(ValueError, match="No JSON"):
            parse_llm_json("No JSON here at all", expect_array=True)

        with pytest.raises(ValueError, match="No JSON"):
            parse_llm_json("No JSON here at all", expect_array=False)

    def test_code_block_without_json_tag(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        text = '```\n{"result": true}\n```'
        result = parse_llm_json(text, expect_array=False)
        assert result["result"] is True

    def test_mixed_text_with_json(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        text = """I found the following segments:

```json
[
  {"clip_id": 1, "start_time": 0, "end_time": 5, "object": "cup", "action": "pick"},
  {"clip_id": 2, "start_time": 5, "end_time": 10, "object": "plate", "action": "place"}
]
```

These are the action segments I identified."""
        result = parse_llm_json(text, expect_array=True)
        assert len(result) == 2
        assert result[0]["object"] == "cup"
        assert result[1]["action"] == "place"

    def test_malformed_json_raises_valueerror(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        # Incomplete JSON that doesn't form a valid array
        text = '[{"broken": true'  # Missing closing brackets
        with pytest.raises(ValueError, match="No JSON"):
            parse_llm_json(text, expect_array=True)

        # JSON that is found but is malformed (has matching brackets but bad content)
        text2 = '[{"broken": true,}]'  # Trailing comma
        with pytest.raises(ValueError, match="malformed"):
            parse_llm_json(text2, expect_array=True)

    def test_nested_json_object(self):
        from video_ingestion_agent.utils.parsing import parse_llm_json

        text = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = parse_llm_json(text, expect_array=False)
        assert result["outer"]["inner"] == [1, 2, 3]
        assert result["flag"] is True

    def test_expect_array_but_code_block_has_object_falls_to_raw(self):
        """If code block has wrong type, fall through to raw search."""
        from video_ingestion_agent.utils.parsing import parse_llm_json

        # Code block has an object, but we expect array; raw text has array
        text = '```json\n{"x": 1}\n```\nAlso: [{"y": 2}]'
        result = parse_llm_json(text, expect_array=True)
        assert isinstance(result, list)
        assert result[0]["y"] == 2


# ========================================================================
# 2. common.parsing.parse_timestamp
# ========================================================================


class TestParseTimestamp:
    """Tests for the shared parse_timestamp utility."""

    def test_float_value(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        assert parse_timestamp(5.5) == 5.5

    def test_int_value(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        assert parse_timestamp(10) == 10.0

    def test_float_string(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        assert parse_timestamp("12.5") == 12.5

    def test_mmss_format(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        assert parse_timestamp("1:30") == 90.0
        assert parse_timestamp("0:05") == 5.0
        assert parse_timestamp("2:00") == 120.0

    def test_hhmmss_format(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        assert parse_timestamp("1:00:00") == 3600.0
        assert parse_timestamp("0:02:30") == 150.0
        assert parse_timestamp("1:30:45") == 5445.0

    def test_zero(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        assert parse_timestamp(0) == 0.0
        assert parse_timestamp("0") == 0.0
        assert parse_timestamp("0:00") == 0.0

    def test_invalid_type_raises(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            parse_timestamp([1, 2, 3])

    def test_invalid_string_raises(self):
        from video_ingestion_agent.utils.parsing import parse_timestamp

        with pytest.raises(ValueError):
            parse_timestamp("not_a_number")


# ========================================================================
# 3. common.video_utils.get_video_info
# ========================================================================


class TestGetVideoInfo:
    """Tests for the shared get_video_info utility."""

    def test_get_video_info_returns_dict(self, tmp_path):
        """Test with a synthetic video created by ffmpeg."""
        from video_ingestion_agent.utils.video_utils import get_video_info

        video_path = tmp_path / "test.mp4"
        # Create a 2-second synthetic video at 10fps
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=10:d=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available or failed to create test video")

        info = get_video_info(video_path)
        assert "frame_count" in info
        assert "fps" in info
        assert "duration" in info
        assert "width" in info
        assert "height" in info
        assert info["fps"] == pytest.approx(10.0, abs=0.5)
        assert info["duration"] == pytest.approx(2.0, abs=0.3)
        assert info["width"] == 64
        assert info["height"] == 64

    def test_get_video_info_nonexistent_raises(self):
        from video_ingestion_agent.utils.video_utils import get_video_info

        with pytest.raises(RuntimeError, match="Could not open video"):
            get_video_info("/nonexistent/video.mp4")


# ========================================================================
# 4. common.video_utils.extract_clip_ffmpeg
# ========================================================================


class TestExtractClipFfmpeg:
    """Tests for the shared extract_clip_ffmpeg utility."""

    def _create_test_video(self, path: Path, duration: float = 4.0) -> bool:
        """Helper to create a synthetic test video."""
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s=64x64:r=10:d={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        return result.returncode == 0

    def test_extract_clip(self, tmp_path):
        from video_ingestion_agent.utils.video_utils import extract_clip_ffmpeg

        video_path = tmp_path / "source.mp4"
        if not self._create_test_video(video_path, duration=4.0):
            pytest.skip("ffmpeg not available")

        output_path = tmp_path / "clip.mp4"
        success = extract_clip_ffmpeg(
            video_path=video_path,
            start_t=1.0,
            end_t=3.0,
            output_path=output_path,
        )
        assert success is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_extract_clip_with_target_fps(self, tmp_path):
        from video_ingestion_agent.utils.video_utils import (
            extract_clip_ffmpeg,
            get_video_info,
        )

        video_path = tmp_path / "source.mp4"
        if not self._create_test_video(video_path, duration=4.0):
            pytest.skip("ffmpeg not available")

        output_path = tmp_path / "clip_fps.mp4"
        success = extract_clip_ffmpeg(
            video_path=video_path,
            start_t=0.0,
            end_t=4.0,
            output_path=output_path,
            target_fps=2,
        )
        assert success is True
        info = get_video_info(output_path)
        assert info["fps"] == pytest.approx(2.0, abs=0.5)

    def test_extract_clip_creates_parent_dirs(self, tmp_path):
        from video_ingestion_agent.utils.video_utils import extract_clip_ffmpeg

        video_path = tmp_path / "source.mp4"
        if not self._create_test_video(video_path):
            pytest.skip("ffmpeg not available")

        output_path = tmp_path / "sub" / "dir" / "clip.mp4"
        success = extract_clip_ffmpeg(
            video_path=video_path,
            start_t=0.0,
            end_t=2.0,
            output_path=output_path,
        )
        assert success is True
        assert output_path.exists()

    def test_extract_clip_nonexistent_source(self, tmp_path):
        from video_ingestion_agent.utils.video_utils import extract_clip_ffmpeg

        success = extract_clip_ffmpeg(
            video_path="/nonexistent/video.mp4",
            start_t=0.0,
            end_t=1.0,
            output_path=tmp_path / "out.mp4",
        )
        assert success is False

    def test_custom_preset_and_crf(self, tmp_path):
        from video_ingestion_agent.utils.video_utils import extract_clip_ffmpeg

        video_path = tmp_path / "source.mp4"
        if not self._create_test_video(video_path):
            pytest.skip("ffmpeg not available")

        output_path = tmp_path / "clip_custom.mp4"
        success = extract_clip_ffmpeg(
            video_path=video_path,
            start_t=0.0,
            end_t=2.0,
            output_path=output_path,
            preset="ultrafast",
            crf=23,
        )
        assert success is True
        assert output_path.exists()


# ========================================================================
# 5. Backward compatibility: segmenter wrappers
# ========================================================================


class TestSegmenterBackwardCompat:
    """Verify that the segmenter module's wrapper functions still work."""

    def test_parse_json_response_delegates(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_json_response

        text = '[{"clip_id": 1, "start_time": 0, "end_time": 5}]'
        result = parse_json_response(text)
        assert len(result) == 1
        assert result[0]["clip_id"] == 1

    def test_parse_json_response_code_block(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_json_response

        text = '```json\n[{"a": 1}]\n```'
        result = parse_json_response(text)
        assert result[0]["a"] == 1

    def test_parse_json_response_no_json(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_json_response

        with pytest.raises(ValueError):
            parse_json_response("no json here")

    def test_parse_timestamp_delegates(self):
        from video_ingestion_agent.ingestion.segmentation.segmenter import parse_timestamp

        assert parse_timestamp(5.0) == 5.0
        assert parse_timestamp("1:30") == 90.0
        assert parse_timestamp("1:00:00") == 3600.0

    def test_get_video_info_returns_tuple(self, tmp_path):
        """get_video_info in segmenter returns (frames, fps, duration) tuple."""
        from video_ingestion_agent.ingestion.segmentation.segmenter import get_video_info

        video_path = tmp_path / "test.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=32x32:r=10:d=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available")

        total_frames, fps, duration = get_video_info(video_path)
        assert isinstance(total_frames, int)
        assert fps == pytest.approx(10.0, abs=0.5)
        assert duration == pytest.approx(1.0, abs=0.3)


# ========================================================================
# 6. Backward compatibility: critic wrappers
# ========================================================================


class TestCriticBackwardCompat:
    """Verify that the critic module's parse_critic_response still works."""

    def test_parse_critic_response_code_block(self):
        from video_ingestion_agent.ingestion.segmentation.critic import parse_critic_response

        text = '```json\n{"is_correct": true, "confidence": 0.95}\n```'
        result = parse_critic_response(text)
        assert result["is_correct"] is True
        assert result["confidence"] == 0.95

    def test_parse_critic_response_raw(self):
        from video_ingestion_agent.ingestion.segmentation.critic import parse_critic_response

        text = 'Analysis: {"is_correct": false, "issues": ["wrong object"]}'
        result = parse_critic_response(text)
        assert result["is_correct"] is False
        assert "wrong object" in result["issues"]

    def test_parse_critic_response_no_json(self):
        from video_ingestion_agent.ingestion.segmentation.critic import parse_critic_response

        with pytest.raises(ValueError):
            parse_critic_response("No JSON here")


# ========================================================================
# 7. Backward compatibility: video_utils re-export
# ========================================================================


class TestVideoUtilsBackwardCompat:
    """Verify that ingestion.video_utils still exports extract_clip_ffmpeg."""

    def test_extract_clip_ffmpeg_importable(self):
        from video_ingestion_agent.ingestion.segmentation.video_utils import extract_clip_ffmpeg

        assert callable(extract_clip_ffmpeg)

    def test_extract_video_chunk_delegates(self, tmp_path):
        from video_ingestion_agent.ingestion.segmentation.video_utils import extract_video_chunk

        video_path = tmp_path / "source.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:r=10:d=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available")

        output_path = tmp_path / "chunk.mp4"
        success = extract_video_chunk(str(video_path), 0.0, 1.0, str(output_path))
        assert success is True
        assert output_path.exists()


# ========================================================================
# 8. Backward compatibility: agents/nodes/base._parse_json
# ========================================================================


class TestBaseNodeParseJson:
    """Verify that BaseNode._parse_json still works after refactoring."""

    def _make_node(self):
        """Create a minimal BaseNode subclass for testing."""
        from video_ingestion_agent.retrieval.config import RetrievalConfig
        from video_ingestion_agent.retrieval.nodes.base import BaseNode

        class DummyNode(BaseNode):
            def __call__(self, state):
                return {}

        return DummyNode(config=RetrievalConfig(), tools={})

    def test_parse_json_code_block(self):
        node = self._make_node()
        text = '```json\n{"key": "value"}\n```'
        result = node._parse_json(text)
        assert result["key"] == "value"

    def test_parse_json_raw(self):
        node = self._make_node()
        text = 'Here is the output: {"answer": 42}'
        result = node._parse_json(text)
        assert result["answer"] == 42

    def test_parse_json_no_json_returns_empty(self):
        node = self._make_node()
        result = node._parse_json("no json here")
        assert result == {}


# ========================================================================
# 9. common module imports
# ========================================================================


class TestCommonImports:
    """Verify all new shared utilities are importable from common package."""

    def test_import_parse_llm_json(self):
        from video_ingestion_agent.utils import parse_llm_json

        assert callable(parse_llm_json)

    def test_import_parse_timestamp(self):
        from video_ingestion_agent.utils import parse_timestamp

        assert callable(parse_timestamp)

    def test_import_get_video_info(self):
        from video_ingestion_agent.utils import get_video_info

        assert callable(get_video_info)

    def test_import_extract_frames_base64(self):
        from video_ingestion_agent.utils import extract_frames_base64

        assert callable(extract_frames_base64)

    def test_import_extract_clip_ffmpeg(self):
        from video_ingestion_agent.utils import extract_clip_ffmpeg

        assert callable(extract_clip_ffmpeg)

    def test_import_model_manager(self):
        from video_ingestion_agent.models import get_model_manager

        assert callable(get_model_manager)


# ========================================================================
# 10. extract_frames_base64 (unit test with mock video)
# ========================================================================


class TestExtractFramesBase64:
    """Tests for the shared extract_frames_base64 utility."""

    def test_extract_frames_from_synthetic_video(self, tmp_path):
        """Create a real short video and extract frames."""
        from video_ingestion_agent.utils.video_utils import extract_frames_base64

        video_path = tmp_path / "test.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=32x32:r=10:d=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available")

        frames = extract_frames_base64(str(video_path), fps=2)
        assert len(frames) > 0
        # Each frame should be a valid base64 string
        import base64

        for f in frames:
            decoded = base64.b64decode(f)
            # JPEG files start with FF D8
            assert decoded[:2] == b"\xff\xd8"

    def test_extract_frames_nonexistent_raises(self):
        from video_ingestion_agent.utils.video_utils import extract_frames_base64

        with pytest.raises(ValueError, match="Could not open video"):
            extract_frames_base64("/nonexistent/video.mp4")


# ========================================================================
# 11. VLLMModel and APIModel delegate to shared utilities
# ========================================================================


class TestModelDelegation:
    """Verify VLLMModel and APIModel delegate to shared implementations."""

    def test_vllm_extract_frames_uses_shared(self):
        """VLLMModel._extract_frames_base64 should delegate to shared."""
        from video_ingestion_agent.models.vllm_model import VLLMModel

        with patch("video_ingestion_agent.models.vllm_model._extract_frames_base64_shared") as mock:
            mock.return_value = ["frame1_b64", "frame2_b64"]

            # Create a mock instance without connecting to server
            model = object.__new__(VLLMModel)
            model.fps = 4
            result = model._extract_frames_base64("test.mp4")

            mock.assert_called_once_with("test.mp4", fps=4)
            assert result == ["frame1_b64", "frame2_b64"]

    def test_api_extract_frames_uses_shared(self):
        """APIModel._extract_frames_base64 should delegate to shared."""
        from video_ingestion_agent.models.api_model import APIModel

        with patch("video_ingestion_agent.models.api_model._extract_frames_base64_shared") as mock:
            mock.return_value = ["frame_b64"]

            model = object.__new__(APIModel)
            model.fps = 2
            result = model._extract_frames_base64("video.mp4")

            mock.assert_called_once_with("video.mp4", fps=2)
            assert result == ["frame_b64"]

    def test_api_get_video_info_uses_shared(self):
        """APIModel._get_video_info should delegate to shared."""
        from video_ingestion_agent.models.api_model import APIModel

        with patch("video_ingestion_agent.models.api_model._get_video_info_shared") as mock:
            mock.return_value = {
                "frame_count": 100,
                "fps": 30.0,
                "duration": 3.33,
                "width": 640,
                "height": 480,
            }

            model = object.__new__(APIModel)
            result = model._get_video_info("video.mp4")

            mock.assert_called_once_with("video.mp4")
            assert result["fps"] == 30.0
            assert result["duration"] == 3.33
