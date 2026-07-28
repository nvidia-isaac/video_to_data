from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPOSITORY_ROOT
    / "reconstruction"
    / "modules"
    / "v2d_wilor"
    / "lib"
    / "video_to_hands.py"
)


def _load_subject(monkeypatch: pytest.MonkeyPatch):
    v2d = types.ModuleType("v2d")
    v2d.__path__ = []
    wilor_package = types.ModuleType("v2d.wilor")
    wilor_package.__path__ = []
    package = types.ModuleType("v2d.wilor.lib")
    package.__path__ = [str(MODULE_PATH.parent)]
    wilor = types.ModuleType("v2d.wilor.lib._wilor")
    wilor.__file__ = str(MODULE_PATH.parent / "_wilor.py")
    wilor._PUBLIC_ARTIFACT_SHA256 = {
        "mano_mean_params.npz": "a" * 64,
        "wilor_final.ckpt": "b" * 64,
        "detector.pt": "c" * 64,
    }
    wilor._validate_weights = lambda _: None
    wilor.get_pipeline = lambda _: object()
    wilor.run_wilor_detect = lambda *_args, **_kwargs: [{"score": 1.0}]
    wilor.run_wilor_on_bboxes = lambda *_args, **_kwargs: [{"score": 1.0}]
    image_to_hands = types.ModuleType("v2d.wilor.lib.image_to_hands")
    image_to_hands.__file__ = str(MODULE_PATH.parent / "image_to_hands.py")
    image_to_hands._load_external_bboxes = lambda _: ([], [])
    package._wilor = wilor
    package.image_to_hands = image_to_hands
    monkeypatch.setitem(sys.modules, "v2d", v2d)
    monkeypatch.setitem(sys.modules, "v2d.wilor", wilor_package)
    monkeypatch.setitem(sys.modules, "v2d.wilor.lib", package)
    monkeypatch.setitem(sys.modules, "v2d.wilor.lib._wilor", wilor)
    monkeypatch.setitem(sys.modules, "v2d.wilor.lib.image_to_hands", image_to_hands)
    name = "_strict_wilor_video_to_hands_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-generation")
    weights = tmp_path / "weights"
    pretrained = weights / "pretrained_models"
    pretrained.mkdir(parents=True)
    for name in (
        "MANO_RIGHT.pkl",
        "mano_mean_params.npz",
        "wilor_final.ckpt",
        "detector.pt",
    ):
        (pretrained / name).write_bytes(name.encode())
    return video, weights


def _fake_decode(_video: str, frames_dir: str) -> int:
    for index in (1, 2):
        Image.fromarray(np.full((4, 5, 3), index, dtype=np.uint8)).save(
            Path(frames_dir) / f"{index:06d}.png"
        )
    return 2


def test_atomic_generation_and_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = _load_subject(monkeypatch)
    video, weights = _inputs(tmp_path)
    output = tmp_path / "raw"
    image_id = "sha256:" + "a" * 64
    monkeypatch.setattr(subject, "_decode_video_to_frames", _fake_decode)
    calls = {"count": 0}

    def detect(*_args, **_kwargs):
        calls["count"] += 1
        return [{"score": 0.5}]

    monkeypatch.setattr(subject, "run_wilor_detect", detect)
    first = subject.video_to_hands(
        str(video), str(output), str(weights), image_id=image_id
    )
    assert first["state"] == "complete"
    assert first["execution_environment"]["container_image_id"] == image_id
    assert first["expected_frames"]["filenames"] == [
        "000000.json",
        "000001.json",
    ]
    assert set(path.name for path in output.iterdir()) == {
        "000000.json",
        "000001.json",
        subject.RUN_GENERATION_FILENAME,
    }
    assert calls["count"] == 2

    resumed = subject.video_to_hands(
        str(video), str(output), str(weights), image_id=image_id
    )
    assert resumed["generation_id"] == first["generation_id"]
    assert calls["count"] == 2

    (output / "stale.json").write_text("[]")
    with pytest.raises(RuntimeError, match="exactly the committed frame set"):
        subject.video_to_hands(str(video), str(output), str(weights), image_id=image_id)


def test_refuses_legacy_directory_and_midrun_input_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = _load_subject(monkeypatch)
    video, weights = _inputs(tmp_path)
    image_id = "sha256:" + "b" * 64
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "000000.json").write_text("[]")
    with pytest.raises(FileExistsError, match="no run_generation.json"):
        subject.video_to_hands(str(video), str(legacy), str(weights), image_id=image_id)

    monkeypatch.setattr(subject, "_decode_video_to_frames", _fake_decode)
    mutated = {"done": False}

    def mutate(*_args, **_kwargs):
        if not mutated["done"]:
            video.write_bytes(b"changed-during-inference")
            mutated["done"] = True
        return [{"score": 1.0}]

    monkeypatch.setattr(subject, "run_wilor_detect", mutate)
    output = tmp_path / "new"
    with pytest.raises(RuntimeError, match="changed during inference"):
        subject.video_to_hands(str(video), str(output), str(weights), image_id=image_id)
    assert not output.exists()
    assert not list(tmp_path.glob(".new.*.partial"))


def test_resume_rejects_tampered_top_level_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = _load_subject(monkeypatch)
    video, weights = _inputs(tmp_path)
    output = tmp_path / "raw"
    image_id = "sha256:" + "d" * 64
    monkeypatch.setattr(subject, "_decode_video_to_frames", _fake_decode)
    subject.video_to_hands(
        str(video), str(output), str(weights), image_id=image_id
    )

    manifest_path = output / subject.RUN_GENERATION_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_environment"]["container_image_id"] = (
        "sha256:" + "e" * 64
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="top-level provenance is inconsistent"):
        subject.video_to_hands(
            str(video), str(output), str(weights), image_id=image_id
        )


def test_image_id_is_mandatory_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = _load_subject(monkeypatch)
    video, weights = _inputs(tmp_path)
    with pytest.raises(ValueError, match="immutable sha256"):
        subject.video_to_hands(
            str(video), str(tmp_path / "raw"), str(weights), image_id="latest"
        )
