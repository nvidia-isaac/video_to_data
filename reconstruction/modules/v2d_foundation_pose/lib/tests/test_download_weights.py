# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess

import pytest

from v2d.foundation_pose.lib import download_weights


def test_gdown_folder_retries_transient_failures(monkeypatch, tmp_path):
    calls = []
    sleeps = []

    def fake_run(command, check):
        calls.append((command, check))
        if len(calls) < download_weights.GDOWN_MAX_ATTEMPTS:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(download_weights.subprocess, "run", fake_run)
    monkeypatch.setattr(download_weights.time, "sleep", sleeps.append)

    download_weights._gdown_folder("folder-id", str(tmp_path))

    expected_command = [
        "gdown", "--folder",
        "https://drive.google.com/drive/folders/folder-id",
        "-O", str(tmp_path),
    ]
    assert calls == [
        (expected_command, True),
        (expected_command, True),
        (expected_command, True),
    ]
    assert sleeps == [
        download_weights.GDOWN_RETRY_DELAY_SECONDS,
        download_weights.GDOWN_RETRY_DELAY_SECONDS,
    ]


def test_gdown_folder_raises_after_final_attempt(monkeypatch, tmp_path):
    error = subprocess.CalledProcessError(1, ["gdown"])
    calls = []
    sleeps = []

    def fake_run(command, check):
        calls.append((command, check))
        raise error

    monkeypatch.setattr(download_weights.subprocess, "run", fake_run)
    monkeypatch.setattr(download_weights.time, "sleep", sleeps.append)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        download_weights._gdown_folder("folder-id", str(tmp_path))

    assert exc_info.value is error
    assert len(calls) == download_weights.GDOWN_MAX_ATTEMPTS
    assert len(sleeps) == download_weights.GDOWN_MAX_ATTEMPTS - 1
