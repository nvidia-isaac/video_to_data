# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parents[1] / "download_weights.py"
_SPEC = importlib.util.spec_from_file_location(
    "foundation_pose_download_weights_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
download_weights = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download_weights)


def _expected(data: bytes):
    return download_weights.WeightFile(len(data), hashlib.sha256(data).hexdigest())


@pytest.fixture
def tiny_weights(monkeypatch):
    expected = {
        download_weights.SCORER_RUN_NAME: {
            "config.yml": _expected(b"scorer config"),
            "model_best.pth": _expected(b"scorer model"),
        },
        download_weights.REFINER_RUN_NAME: {
            "config.yml": _expected(b"refiner config"),
            "model_best.pth": _expected(b"refiner model"),
        },
    }
    monkeypatch.setattr(download_weights, "EXPECTED_WEIGHTS", expected)
    return expected


def _write_run(root: Path, run_name: str, expected) -> None:
    directory = root / run_name
    directory.mkdir(parents=True)
    for filename in expected[run_name]:
        if run_name == download_weights.SCORER_RUN_NAME:
            kind = "scorer"
        else:
            kind = "refiner"
        content = f"{kind} {'config' if filename == 'config.yml' else 'model'}".encode()
        (directory / filename).write_bytes(content)


def test_manifest_contains_exact_published_foundationpose_artifacts() -> None:
    expected = download_weights.EXPECTED_WEIGHTS
    assert expected[download_weights.SCORER_RUN_NAME] == {
        "config.yml": download_weights.WeightFile(
            778,
            "a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f",
        ),
        "model_best.pth": download_weights.WeightFile(
            190_229_389,
            "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
        ),
    }
    assert expected[download_weights.REFINER_RUN_NAME] == {
        "config.yml": download_weights.WeightFile(
            708,
            "28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686",
        ),
        "model_best.pth": download_weights.WeightFile(
            68_220_109,
            "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
        ),
    }


def test_exact_existing_files_skip_every_download(
    tmp_path: Path, monkeypatch, tiny_weights
) -> None:
    for run_name in tiny_weights:
        _write_run(tmp_path, run_name, tiny_weights)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("valid weights must not launch gdown")

    monkeypatch.setattr(download_weights.subprocess, "run", unexpected_run)
    download_weights.download_weights(str(tmp_path))
    download_weights.verify_weights(tmp_path)


@pytest.mark.parametrize("corruption", ["missing", "wrong_size", "wrong_hash"])
def test_corrupt_expected_file_is_replaced_from_verified_staging(
    tmp_path: Path, monkeypatch, tiny_weights, corruption: str
) -> None:
    for run_name in tiny_weights:
        _write_run(tmp_path, run_name, tiny_weights)

    target = tmp_path / download_weights.SCORER_RUN_NAME / "config.yml"
    if corruption == "missing":
        target.unlink()
    elif corruption == "wrong_size":
        target.write_bytes(b"x")
    else:
        target.write_bytes(b"SCORER CONFIG")

    calls: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        calls.append(command)
        destination = Path(command[command.index("-O") + 1])
        run_name = destination.name
        _write_run(destination.parent, run_name, tiny_weights)

    monkeypatch.setattr(download_weights.subprocess, "run", fake_run)
    download_weights.download_weights(str(tmp_path))

    assert len(calls) == 1
    assert download_weights.SCORER_FOLDER_ID in calls[0][2]
    download_weights.verify_weights(tmp_path)
    assert not list(tmp_path.rglob("*.partial"))


def test_unverified_download_never_replaces_existing_file(
    tmp_path: Path, monkeypatch, tiny_weights
) -> None:
    for run_name in tiny_weights:
        _write_run(tmp_path, run_name, tiny_weights)
    target = tmp_path / download_weights.SCORER_RUN_NAME / "config.yml"
    target.write_bytes(b"corrupt existing")

    def fake_run(command, *, check):
        destination = Path(command[command.index("-O") + 1])
        destination.mkdir(parents=True)
        (destination / "config.yml").write_bytes(b"untrusted")
        (destination / "model_best.pth").write_bytes(b"scorer model")

    monkeypatch.setattr(download_weights.subprocess, "run", fake_run)
    with pytest.raises(download_weights.WeightIntegrityError):
        download_weights.download_weights(str(tmp_path))

    assert target.read_bytes() == b"corrupt existing"
    assert not list(tmp_path.rglob("*.partial"))
