# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Unit tests for the webapp's reconstruction orchestrator.

The heavy work (Docker containers, weights, GPU) is mocked out — these tests
only validate the orchestration logic: argv shape, marker parsing, error
propagation, and config validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_ingestion_agent.webapp.models.reconstruction import (
    STAGES,
    ReconstructionRequest,
)
from video_ingestion_agent.webapp.services.reconstruction_service import (
    ReconstructionConfig,
    ReconstructionService,
)


@pytest.fixture
def example_request() -> ReconstructionRequest:
    return ReconstructionRequest.from_clip_dict(
        {
            "clip_id": "test_clip_0001",
            "video_path": "/data/test.mp4",
            "start_t": 0.0,
            "end_t": 1.5,
            "object": "wooden bowl",
            "action": "picking up",
            "description": "test",
        }
    )


@pytest.fixture
def cfg(tmp_path: Path) -> ReconstructionConfig:
    weights = tmp_path / "weights"
    for sub in ("moge", "gd", "sam2", "sam3d", "fp", "hand"):
        (weights / sub).mkdir(parents=True)
    fake_python = tmp_path / "fake_python"
    fake_python.write_text("#!/bin/sh\n")
    fake_recon = tmp_path / "fake_reconstruction"
    fake_recon.mkdir()
    return ReconstructionConfig(
        out_root=tmp_path / "out",
        moge_weights=weights / "moge",
        grounding_dino_weights=weights / "gd",
        sam2_weights=weights / "sam2",
        sam3d_weights=weights / "sam3d",
        foundation_pose_weights=weights / "fp",
        hand_reconstruction_weights=weights / "hand",
        reconstruction_python=fake_python,
        reconstruction_root=fake_recon,
    )


def _fake_proc(stdout_lines: list[str], returncode: int = 0) -> MagicMock:
    """Build a Popen-shaped mock that yields the given stdout lines."""
    proc = MagicMock()
    proc.stdout = iter(line + "\n" for line in stdout_lines)
    proc.returncode = returncode
    proc.wait.return_value = returncode
    return proc


def test_request_from_clip_dict():
    req = ReconstructionRequest.from_clip_dict(
        {
            "clip_id": "abc",
            "video_path": "/v.mp4",
            "start_t": 1.0,
            "end_t": 2.5,
            "object": "mug",
        }
    )
    # segment_id prefixes the source-video stem so clip_ids that recur
    # across videos don't collide in the recon out_root.
    assert req.segment_id == "v__abc"
    assert req.start_t == 1.0
    assert req.end_t == 2.5
    assert req.object_label == "mug"
    assert req.ref_frame == 0
    assert req.object_id == 1


def test_request_segment_id_doesnt_double_prefix():
    """If clip_id already starts with the video stem, don't double-prefix."""
    req = ReconstructionRequest.from_clip_dict(
        {
            "clip_id": "T1_C1_clip_0001",
            "video_path": "/data/T1_C1.mp4",
            "start_t": 0.0,
            "end_t": 1.0,
            "object": "mug",
        }
    )
    assert req.segment_id == "T1_C1_clip_0001"


def test_request_from_query_result_clip_shape():
    """QueryResult.clips uses start_time/end_time and has no clip_id —
    the reader synthesizes a deterministic segment_id from video stem + range."""
    req = ReconstructionRequest.from_clip_dict(
        {
            "video_path": "/data/foo/bar.mp4",
            "start_time": 1.3,
            "end_time": 8.5,
            "object": "spoon",
            "action": "moving",
        }
    )
    assert req.start_t == 1.3
    assert req.end_t == 8.5
    assert req.object_label == "spoon"
    assert req.action_label == "moving"
    # segment_id is synthesized — must be deterministic and filesystem-safe.
    assert req.segment_id == "bar_1_30s-8_50s"


def test_validate_reports_missing_weights():
    cfg = ReconstructionConfig()
    issues = cfg.validate()
    assert any("moge weights path not configured" in i for i in issues)


def test_segment_jsonl_round_trip(cfg, example_request):
    svc = ReconstructionService(cfg)
    seg_jsonl = svc._write_segment_jsonl(example_request)
    try:
        line = seg_jsonl.read_text().strip()
        row = json.loads(line)
        assert row["clip_id"] == example_request.segment_id
        assert row["video_path"] == str(example_request.video_path)
        assert row["start_t"] == example_request.start_t
    finally:
        seg_jsonl.unlink(missing_ok=True)


def test_collect_result_finds_artifacts(cfg, example_request, tmp_path):
    svc = ReconstructionService(cfg)
    seg_dir = cfg.out_root / example_request.segment_id
    seg_dir.mkdir(parents=True)
    # Full-mode artifact filenames carry the depth_source suffix.
    (seg_dir / "render_aligned_moge.mp4").write_bytes(b"fake")
    (seg_dir / "mesh_scaled_moge.obj").write_bytes(b"fake")

    result = svc.collect_result(example_request)
    assert result.render_mp4 == seg_dir / "render_aligned_moge.mp4"
    assert result.scaled_mesh == seg_dir / "mesh_scaled_moge.obj"


_STAGE_MODULE = "video_ingestion_agent.reconstruction_interface.ego_e2e.run_ego_e2e"


def test_build_cmd(cfg, example_request):
    """argv shape includes python -m <module> --segments … --depth-source …"""
    svc = ReconstructionService(cfg)
    seg_jsonl = svc._write_segment_jsonl(example_request)
    try:
        cmd = svc._build_cmd(example_request, seg_jsonl)
        assert cmd[1] == "-m"
        assert cmd[2] == _STAGE_MODULE
        assert "--segments" in cmd
        assert "--out" in cmd
        assert "--reconstruction-root" in cmd
        assert "--reconstruction-python" in cmd
        assert "--moge-weights" in cmd
        assert "--hand-reconstruction-weights" in cmd
        assert "--depth-source" in cmd
        assert "moge" in cmd  # default depth source
    finally:
        seg_jsonl.unlink(missing_ok=True)


def test_run_parses_step_markers(cfg, example_request):
    """`[run ] <label>` lines from the orchestrator map to StageEvents on the
    matching stage_id, in label order."""
    svc = ReconstructionService(cfg)
    # A subset of marker lines covering early, middle, and late stages —
    # `(moge)` / `(moge depth)` suffixes are baked into some labels by the
    # orchestrator, so the parser must use startswith.
    fake_stdout = [
        "  [run ] Ego hand reconstruction (ViPE + Dyn-HaMR)",
        "INFO: ViPE running …",  # non-marker — falls through as a log line
        "  [run ] Extract frames",
        "  [run ] SAM3D mesh generation (moge depth)",
        "  [run ] Render aligned (trans_aligned, moge)",
    ]
    fake = _fake_proc(fake_stdout, returncode=0)
    with patch(
        "video_ingestion_agent.webapp.services.reconstruction_service.subprocess.Popen",
        return_value=fake,
    ):
        events = list(svc.run(example_request))

    # Each marker should produce a 'running' event tagged with the matching
    # stage_id. We check membership rather than exact equality because
    # non-marker lines also produce running events (passed through to the log
    # viewer).
    running_stages = {e.stage for e in events if e.status == "running"}
    assert "ego_hand_recon" in running_stages
    assert "extract_frames" in running_stages
    assert "sam3d_mesh" in running_stages
    assert "render_aligned" in running_stages

    # The closing event after exit 0 should mark the last seen stage as ok.
    final_ok = [e for e in events if e.status == "ok"]
    assert final_ok, "expected at least one ok event after exit 0"
    assert final_ok[-1].stage in STAGES


def test_run_handles_skip_markers(cfg, example_request):
    """`[skip] <label>` lines emit immediate cached `ok` StageEvents."""
    svc = ReconstructionService(cfg)
    fake_stdout = [
        "  [skip] Ego hand reconstruction (ViPE + Dyn-HaMR)",
        "  [skip] Extract frames",
        "  [run ] MoGe depth + intrinsics",
    ]
    fake = _fake_proc(fake_stdout, returncode=0)
    with patch(
        "video_ingestion_agent.webapp.services.reconstruction_service.subprocess.Popen",
        return_value=fake,
    ):
        events = list(svc.run(example_request))

    # Cached `ok` events for the two skipped stages.
    cached = [e for e in events if e.status == "ok" and "cached" in e.message]
    cached_stages = {e.stage for e in cached}
    assert "ego_hand_recon" in cached_stages
    assert "extract_frames" in cached_stages
    # The non-skipped MoGe stage produces a running event (no cached marker).
    running_with_marker = [e for e in events if e.status == "running" and "MoGe depth" in e.message]
    assert any(e.stage == "moge_depth" for e in running_with_marker)


def test_run_marks_failed_stage(cfg, example_request):
    """Non-zero exit produces an err event tagged with the last seen stage."""
    svc = ReconstructionService(cfg)
    fake_stdout = [
        "  [run ] Ego hand reconstruction (ViPE + Dyn-HaMR)",
        "boom",
    ]
    fake = _fake_proc(fake_stdout, returncode=1)
    with patch(
        "video_ingestion_agent.webapp.services.reconstruction_service.subprocess.Popen",
        return_value=fake,
    ):
        events = list(svc.run(example_request))

    err = [e for e in events if e.status == "err"]
    assert err, "expected at least one err event on non-zero exit"
    assert err[-1].stage == "ego_hand_recon"
    assert "exit 1" in err[-1].message
