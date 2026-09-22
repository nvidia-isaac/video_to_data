# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for compact local training progress."""

import pytest

from flash_chord.training.progress import ConsoleTrainingProgress


def test_console_progress_prints_start_percent_boundaries_and_completion(monkeypatch, capsys, tmp_path):
    times = iter((100.0, 110.0, 120.0))
    monkeypatch.setattr("flash_chord.training.progress.time.perf_counter", lambda: next(times))
    progress = ConsoleTrainingProgress(1_000, tmp_path)

    progress.start(0)
    progress.update(9)
    progress.update(10)
    progress.update(19)
    progress.update(1_000)

    assert capsys.readouterr().out.splitlines() == [
        f"training progress: 0/1,000 environment steps (0.0%); checkpoints: {tmp_path}",
        "training progress: 10/1,000 environment steps (1.0%); 1 env steps/s; elapsed 00:00:10; ETA 00:16:30",
        "training progress: 1,000/1,000 environment steps (100.0%); 50 env steps/s; elapsed 00:00:20; ETA 00:00:00",
    ]


def test_console_progress_handles_resume_and_rejects_invalid_updates(capsys, tmp_path):
    progress = ConsoleTrainingProgress(1_000, tmp_path)
    progress.start(500)
    progress.update(500)
    assert "500/1,000 environment steps (50.0%)" in capsys.readouterr().out

    with pytest.raises(ValueError, match="must not decrease"):
        progress.update(499)
    with pytest.raises(ValueError, match="must be in"):
        progress.update(1_001)


def test_console_progress_requires_positive_total_and_start(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        ConsoleTrainingProgress(0, tmp_path)

    progress = ConsoleTrainingProgress(1, tmp_path)
    with pytest.raises(RuntimeError, match="must be started"):
        progress.update(1)
