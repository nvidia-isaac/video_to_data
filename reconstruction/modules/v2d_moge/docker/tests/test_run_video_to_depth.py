from __future__ import annotations

import json
from pathlib import Path

import pytest

from v2d.moge.docker import run_video_to_depth as runner


IMAGE_ID = "sha256:" + "a" * 64


def _fake_inference(calls: list[dict]):
    def run_in_container(**kwargs) -> None:
        calls.append(kwargs)
        extensions = {
            "depth_folder": ".png",
            "intrinsics_folder": ".json",
            "points_folder": ".npy",
            "normals_folder": ".npy",
            "mask_folder": ".png",
        }
        for output_name, output_path in kwargs["outputs"].items():
            directory = Path(output_path)
            directory.mkdir(parents=True, exist_ok=True)
            extension = extensions[output_name]
            for index in range(2):
                (directory / f"{index:06d}{extension}").write_bytes(
                    f"{output_name}:{index}".encode()
                )

    return run_in_container


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"rgb-video")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    (weights / "model.pt").write_bytes(b"moge-model")
    return video, weights


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    input_intrinsics_path: Path | None = None,
) -> tuple[dict, list[dict]]:
    video, weights = _inputs(tmp_path)
    root = tmp_path / name
    calls: list[dict] = []
    monkeypatch.setattr(runner, "run_in_container", _fake_inference(calls))
    manifest = runner.run_video_to_depth(
        video_path=str(video),
        depth_folder=str(root / "depth"),
        intrinsics_folder=str(root / "intrinsics"),
        weights_path=str(weights),
        input_intrinsics_path=(
            None if input_intrinsics_path is None else str(input_intrinsics_path)
        ),
        mask_folder=str(root / "mask"),
        image_id=IMAGE_ID,
        gpu=1,
    )
    return manifest, calls


def test_rgb_only_generation_commits_exact_null_intrinsics_and_output_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, calls = _run(tmp_path, monkeypatch, name="first")

    assert manifest["state"] == "complete"
    assert manifest["execution_environment"] == {
        "container_image_id": IMAGE_ID
    }
    assert manifest["sources"]["input_intrinsics"] is None
    assert manifest["parameters"]["input_intrinsics_path"] is None
    assert manifest["parameters"]["intrinsics_mode"] == "estimated_from_rgb"
    assert manifest["parameters"]["requested_outputs"] == [
        "depth",
        "intrinsics",
        "mask",
    ]
    assert manifest["expected_frames"] == {"count": 2, "indices": [0, 1]}
    assert calls[0]["image"] == IMAGE_ID
    assert "input_intrinsics_path" not in calls[0]["inputs"]
    assert calls[0]["gpu_device"] == 1

    manifest_path = tmp_path / "first" / "depth" / runner.RUN_GENERATION_FILENAME
    assert json.loads(manifest_path.read_text()) == manifest
    assert runner.validate_generation_manifest(manifest_path) == manifest
    (tmp_path / "first" / "mask" / "000001.png").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="output bytes"):
        runner.validate_generation_manifest(manifest_path)


def test_generation_records_a_supplied_intrinsics_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intrinsics = tmp_path / "known_k.json"
    intrinsics.write_text('{"fx": 1000}')
    manifest, calls = _run(
        tmp_path,
        monkeypatch,
        name="known-k",
        input_intrinsics_path=intrinsics,
    )

    assert manifest["sources"]["input_intrinsics"]["sha256"]
    assert manifest["parameters"]["input_intrinsics_path"] == str(
        intrinsics.resolve()
    )
    assert manifest["parameters"]["intrinsics_mode"] == (
        "known_horizontal_fov_prior"
    )
    assert calls[0]["inputs"]["input_intrinsics_path"] == str(
        intrinsics.resolve()
    )
    assert "input_intrinsics_path" in calls[0]["input_files"]


def test_generation_id_is_path_independent_for_identical_rgb_only_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = _run(tmp_path, monkeypatch, name="a")
    second, _ = _run(tmp_path, monkeypatch, name="b")
    assert first["generation_id"] == second["generation_id"]


def test_refuses_to_mix_with_uncommitted_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, weights = _inputs(tmp_path)
    depth = tmp_path / "output" / "depth"
    depth.mkdir(parents=True)
    (depth / "000000.png").write_bytes(b"stale")
    monkeypatch.setattr(runner, "run_in_container", lambda **_: None)

    with pytest.raises(FileExistsError, match="refusing to mix generations"):
        runner.run_video_to_depth(
            video_path=str(video),
            depth_folder=str(depth),
            intrinsics_folder=str(tmp_path / "output" / "intrinsics"),
            weights_path=str(weights),
            image_id=IMAGE_ID,
        )
