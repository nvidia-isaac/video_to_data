# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Input-mode dispatch for the webapp Ingest tab.

Regression cover for the silent both-populated case: the tab used to test the
batch directory first and return unconditionally, so an uploaded file was
discarded without a message whenever the directory box was non-empty.
"""

import pytest

from video_ingestion_agent.webapp.tabs.ingestion_tab import (
    _AMBIGUOUS_INPUT_MSG,
    _NO_INPUT_MSG,
    resolve_ingestion_mode,
)

VIDEO = "/data/test_videos/P0001_stir_bowl_3s-16s.mp4"
DIRECTORY = "/data/test_videos"


def test_upload_only_runs_single():
    assert resolve_ingestion_mode(VIDEO, "") == ("single", "")


def test_directory_only_runs_batch():
    assert resolve_ingestion_mode(None, DIRECTORY) == ("batch", "")


def test_both_populated_is_refused_not_guessed():
    """The defect: this used to silently return batch and drop the upload."""
    mode, message = resolve_ingestion_mode(VIDEO, DIRECTORY)
    assert mode == "error"
    assert message == _AMBIGUOUS_INPUT_MSG
    # The message has to name both inputs, or it cannot be acted on.
    assert "Batch directory path" in message
    assert "Single file video" in message


def test_neither_populated_is_refused():
    assert resolve_ingestion_mode(None, None) == ("error", _NO_INPUT_MSG)


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
def test_whitespace_directory_is_not_a_directory(blank):
    """A box holding only whitespace must not trigger the ambiguity error."""
    assert resolve_ingestion_mode(VIDEO, blank) == ("single", "")
    assert resolve_ingestion_mode(None, blank) == ("error", _NO_INPUT_MSG)


@pytest.mark.parametrize("empty_upload", [None, "", []])
def test_falsy_uploads_are_not_an_upload(empty_upload):
    """Gradio clears the File component to None; be tolerant of "" and []."""
    assert resolve_ingestion_mode(empty_upload, DIRECTORY) == ("batch", "")


def test_untrimmed_directory_still_dispatches_batch():
    assert resolve_ingestion_mode(None, f"  {DIRECTORY}  ") == ("batch", "")


# ---------------------------------------------------------------------------
# Handler-level cover: the dispatch, not just the decision function.
#
# The two workers are closures inside create_ingestion_tab and cannot be
# patched, so instead we make their observable side effects fatal: the batch
# path shells out via subprocess.Popen, the single-file path constructs an
# IngestionService. If either fires while the form is ambiguous, the test
# fails loudly.
# ---------------------------------------------------------------------------


def _build_handler(monkeypatch, default_videos_dir=""):
    """Build the Ingest tab and return its ``run_ingestion`` handler."""
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    gr = pytest.importorskip("gradio")

    from video_ingestion_agent.webapp.config import AppConfig
    from video_ingestion_agent.webapp.tabs import ingestion_tab as mod

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("batch ingestion was launched")

    monkeypatch.setattr(mod.subprocess, "Popen", _no_subprocess)

    config = AppConfig()
    config.default_videos_dir = default_videos_dir

    with gr.Blocks():
        components = mod.create_ingestion_tab(services={}, config=config)

    return components["run_ingestion"]


def _drive(handler, video_file, videos_dir):
    return list(handler(video_file, videos_dir, None, "outputs/", False, 1, False))


def test_handler_refuses_both_and_launches_nothing(monkeypatch, tmp_path):
    """The reported defect, at the level the user actually hits it.

    Before the fix this ran a batch over *directory* and discarded *upload*,
    emitting a success summary. Now it must emit one error and stop.
    """
    handler = _build_handler(monkeypatch)
    directory = tmp_path / "videos"
    directory.mkdir()
    upload = tmp_path / "clip.mp4"
    upload.write_bytes(b"")

    outputs = _drive(handler, str(upload), str(directory))

    assert len(outputs) == 1, "must stop immediately, not start a run"
    progress_md, _logs, _results_section, results = outputs[0]
    assert "Ambiguous input" in progress_md
    assert "Batch directory path" in progress_md
    assert "Single file video" in progress_md
    assert results is None, "no results payload for a refused run"


def test_handler_refuses_neither(monkeypatch):
    handler = _build_handler(monkeypatch)
    outputs = _drive(handler, None, "")
    assert len(outputs) == 1
    assert outputs[0][0] == _NO_INPUT_MSG
    assert outputs[0][3] is None


def test_handler_ambiguity_survives_a_prefilled_directory(monkeypatch, tmp_path):
    """The OSMO webapp config pre-fills the directory box on a fresh page.

    A user who only ever touched the upload control still hits the conflict,
    so the refusal must not depend on the box having been typed into.
    """
    directory = tmp_path / "videos"
    directory.mkdir()
    handler = _build_handler(monkeypatch, default_videos_dir=str(directory))
    upload = tmp_path / "clip.mp4"
    upload.write_bytes(b"")

    outputs = _drive(handler, str(upload), str(directory))

    assert len(outputs) == 1
    assert "Ambiguous input" in outputs[0][0]
