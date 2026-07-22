# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

from v2d.docker import container


def _capture_command(monkeypatch, **kwargs) -> list[str]:
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command, *, check):
        calls.append((command, check))

    monkeypatch.setattr(container.subprocess, "run", fake_run)
    container.run_in_container(
        image="test-image",
        module="test.module",
        **kwargs,
    )
    assert len(calls) == 1
    command, check = calls[0]
    assert check is True
    return command


def _volumes(command: list[str]) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command) if item == "-v"]


def _argument(command: list[str], name: str) -> str:
    index = command.index(f"--{name}")
    return command[index + 1]


def test_same_parent_coalesces_to_one_rw_alias_in_non_strict_mode(
    tmp_path, monkeypatch
) -> None:
    shared = tmp_path / "shared"
    source = shared / "source.json"
    result = shared / "result.json"

    command = _capture_command(
        monkeypatch,
        inputs={"source": str(source)},
        outputs={"result": str(result)},
    )

    host = os.path.abspath(shared)
    assert _volumes(command) == [f"{host}:/data/source"]
    assert _argument(command, "source") == "/data/source/source.json"
    assert _argument(command, "result") == "/data/source/result.json"


def test_non_strict_shared_root_preserves_host_relative_symlinks(
    tmp_path, monkeypatch
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    output = job / "stage1_recon"

    command = _capture_command(
        monkeypatch,
        inputs={
            "frames_meta": str(job / "frames_meta.json"),
            "depth_dir": str(job / "depth"),
        },
        outputs={"output_dir": str(output)},
    )

    container_source = f'{_argument(command, "depth_dir")}/000123.png'
    container_destination_parent = f'{_argument(command, "output_dir")}/depth'
    stored_relative_link = os.path.relpath(
        container_source, container_destination_parent
    )
    host_resolution = os.path.normpath(
        os.path.join(output, "depth", stored_relative_link)
    )

    assert host_resolution == str(job / "depth" / "000123.png")


def test_exact_input_file_allows_strict_output_beneath_its_parent(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "data"
    video = run_root / "clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    masks = run_root / "outputs" / "clip" / "masks"

    command = _capture_command(
        monkeypatch,
        inputs={"video_path": str(video)},
        outputs={"masks_dir": str(masks)},
        input_files={"video_path"},
        output_directories={"masks_dir"},
        strict_io_isolation=True,
    )

    assert _volumes(command) == [
        f"{os.path.abspath(video)}:/data/video_path/clip.mp4:ro",
        f"{os.path.abspath(masks)}:/data/masks_dir",
    ]
    assert _argument(command, "video_path") == "/data/video_path/clip.mp4"
    assert _argument(command, "masks_dir") == "/data/masks_dir"


def test_exact_input_file_must_exist_before_output_mutation(tmp_path, monkeypatch) -> None:
    output = tmp_path / "outputs" / "masks"

    with pytest.raises(FileNotFoundError, match="exact input files must exist"):
        _capture_command(
            monkeypatch,
            inputs={"video_path": str(tmp_path / "missing.mp4")},
            outputs={"masks_dir": str(output)},
            input_files={"video_path"},
            output_directories={"masks_dir"},
            strict_io_isolation=True,
        )

    assert not output.exists()


def test_strict_io_isolation_rejects_shared_parent_before_host_mutation(
    tmp_path, monkeypatch
) -> None:
    shared = tmp_path / "shared"
    output = shared / "nested" / "result.json"

    with pytest.raises(ValueError, match="host-directory overlap"):
        _capture_command(
            monkeypatch,
            inputs={"source": str(shared / "source.json")},
            outputs={"result": str(output)},
            strict_io_isolation=True,
        )

    assert not output.parent.exists()


def test_distinct_directories_keep_input_ro_output_rw_and_gpu_network_options(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"

    command = _capture_command(
        monkeypatch,
        inputs={
            "source": str(input_dir / "source.mp4"),
            "mask": str(input_dir / "mask.npy"),
        },
        outputs={"result": str(output_dir / "result.mp4")},
        gpus=True,
        gpu_device=2,
        network_disabled=True,
    )

    assert _volumes(command) == [
        f"{os.path.abspath(input_dir)}:/data/source:ro",
        f"{os.path.abspath(output_dir)}:/data/result",
    ]
    assert _argument(command, "source") == "/data/source/source.mp4"
    assert _argument(command, "mask") == "/data/source/mask.npy"
    assert _argument(command, "result") == "/data/result/result.mp4"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--gpus") + 1] == "device=2"
    assert "all" not in command
    assert not any(value.startswith("HOME=") for value in command)


def test_declared_directory_arguments_mount_exact_sibling_roots(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    rgb_dir = run_root / "rgb"
    output_dir = run_root / "masks"
    rgb_dir.mkdir(parents=True)

    command = _capture_command(
        monkeypatch,
        inputs={"rgb_dir": str(rgb_dir)},
        outputs={"output_dir": str(output_dir)},
        input_directories={"rgb_dir"},
        output_directories={"output_dir"},
        strict_io_isolation=True,
    )

    assert _volumes(command) == [
        f"{os.path.abspath(rgb_dir)}:/data/rgb_dir:ro",
        f"{os.path.abspath(output_dir)}:/data/output_dir",
    ]
    assert _argument(command, "rgb_dir") == "/data/rgb_dir"
    assert _argument(command, "output_dir") == "/data/output_dir"
    assert output_dir.is_dir()


def test_directory_argument_names_must_reference_present_paths(
    tmp_path, monkeypatch
) -> None:
    with pytest.raises(ValueError, match="must identify present path arguments"):
        _capture_command(
            monkeypatch,
            inputs={"source": str(tmp_path / "source.json")},
            outputs={},
            input_directories={"typo"},
        )


def test_atomic_output_directory_mounts_parent_without_precreating_leaf(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "run" / "wilor_raw"

    command = _capture_command(
        monkeypatch,
        inputs={},
        outputs={"output_dir": str(output_dir)},
        atomic_output_directories={"output_dir"},
    )

    assert _volumes(command) == [
        f"{os.path.abspath(output_dir.parent)}:/data/output_dir"
    ]
    assert _argument(command, "output_dir") == "/data/output_dir/wilor_raw"
    assert output_dir.parent.is_dir()
    assert not output_dir.exists()


def test_glob_uses_clean_base_dir_and_routes_shared_output_to_rw_mount(
    tmp_path, monkeypatch
) -> None:
    capture_root = tmp_path / "captures"
    image_glob = capture_root / "*" / "frames" / "*.png"
    report = capture_root / "report.json"
    assert container._base_dir(str(image_glob)) == os.path.abspath(capture_root)

    command = _capture_command(
        monkeypatch,
        inputs={"images": str(image_glob)},
        outputs={"report": str(report)},
    )

    host = os.path.abspath(capture_root)
    assert _volumes(command) == [f"{host}:/data/images"]
    assert _argument(command, "images") == "/data/images/*/frames/*.png"
    assert _argument(command, "report") == "/data/images/report.json"


def test_rejects_identical_input_and_output_argument_names(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "input" / "source.json"
    source.parent.mkdir()
    source.write_text("{}")

    with pytest.raises(ValueError, match="path arguments must use distinct names"):
        _capture_command(
            monkeypatch,
            inputs={"data": str(source)},
            outputs={"data": str(tmp_path / "output" / "result.json")},
            input_files={"data"},
        )
