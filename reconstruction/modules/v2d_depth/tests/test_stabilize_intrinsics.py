from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from v2d.depth.lib import stabilize_intrinsics as stabilizer
from v2d.moge.docker import run_video_to_depth as moge_runner


IMAGE_ID = "sha256:" + "a" * 64


def _fake_inference(**kwargs) -> None:
    intrinsics_values = [
        {
            "fx": 1064.0,
            "fy": 1062.0,
            "cx": 951.0,
            "cy": 531.0,
            "width": 1920,
            "height": 1080,
        },
        {
            "fx": 1037.0,
            "fy": 1039.0,
            "cx": 955.0,
            "cy": 535.0,
            "width": 1920,
            "height": 1080,
        },
        {
            "fx": 1011.0,
            "fy": 1013.0,
            "cx": 969.0,
            "cy": 549.0,
            "width": 1920,
            "height": 1080,
        },
    ]
    for output_name, output_path in kwargs["outputs"].items():
        directory = Path(output_path)
        directory.mkdir(parents=True, exist_ok=True)
        for index, intrinsics in enumerate(intrinsics_values):
            if output_name == "intrinsics_folder":
                (directory / f"{index:06d}.json").write_text(
                    json.dumps(intrinsics),
                    encoding="utf-8",
                )
            else:
                (directory / f"{index:06d}.png").write_bytes(
                    f"depth:{index}".encode()
                )


def _moge_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"rgb-only-video")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.pt").write_bytes(b"moge-model")
    root = tmp_path / "moge"
    monkeypatch.setattr(moge_runner, "run_in_container", _fake_inference)
    moge_runner.run_video_to_depth(
        video_path=str(video),
        depth_folder=str(root / "depth"),
        intrinsics_folder=str(root / "intrinsics"),
        weights_path=str(weights),
        image_id=IMAGE_ID,
    )
    return root / "intrinsics", root / "depth" / "run_generation.json"


def test_fixed_principal_point_generation_binds_exact_moge_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intrinsics, moge_manifest = _moge_generation(tmp_path, monkeypatch)
    output = tmp_path / "stable" / "intrinsics_stable.json"

    stable = stabilizer.stabilize_intrinsics_with_provenance(
        intrinsics_folder=intrinsics,
        output_path=output,
        moge_generation_manifest_path=moge_manifest,
        fix_principal_point=True,
    )

    assert stable.to_dict() == {
        "fx": 1037.0,
        "fy": 1039.0,
        "cx": 960.0,
        "cy": 540.0,
        "width": 1920,
        "height": 1080,
    }
    sidecar = output.with_suffix(".generation.json")
    manifest = stabilizer.validate_stable_intrinsics_manifest(sidecar)
    assert manifest["parameters"]["fix_principal_point"] is True
    assert manifest["parameters"]["principal_point_policy"] == "image_center"
    assert manifest["sources"]["moge_generation_id"].startswith("sha256:")
    assert manifest["sources"]["moge_generation_manifest"]["sha256"]
    assert manifest["sources"]["intrinsics"]["file_count"] == 3
    assert set(manifest["sources"]["intrinsics"]["files"]) == {
        "000000.json",
        "000001.json",
        "000002.json",
    }
    assert manifest["output"]["stable_intrinsics"]["sha256"]
    assert manifest["generation_id"].startswith("sha256:")


def test_estimated_principal_point_policy_uses_temporal_median(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intrinsics, moge_manifest = _moge_generation(tmp_path, monkeypatch)
    output = tmp_path / "stable.json"
    sidecar = tmp_path / "stable.provenance.json"

    stable = stabilizer.stabilize_intrinsics_with_provenance(
        intrinsics,
        output,
        moge_manifest,
        provenance_manifest_path=sidecar,
    )

    assert stable.cx == 955.0
    assert stable.cy == 535.0
    manifest = stabilizer.validate_stable_intrinsics_manifest(sidecar)
    assert manifest["parameters"]["principal_point_policy"] == "temporal_median"


def test_rejects_identical_bytes_from_a_directory_not_committed_by_moge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intrinsics, moge_manifest = _moge_generation(tmp_path, monkeypatch)
    copied = tmp_path / "copied_intrinsics"
    shutil.copytree(intrinsics, copied)

    with pytest.raises(RuntimeError, match="exact directory"):
        stabilizer.stabilize_intrinsics_with_provenance(
            copied,
            tmp_path / "stable.json",
            moge_manifest,
            fix_principal_point=True,
        )


def test_validator_rejects_changed_intrinsics_and_changed_stable_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intrinsics, moge_manifest = _moge_generation(tmp_path, monkeypatch)
    output = tmp_path / "stable.json"
    stabilizer.stabilize_intrinsics_with_provenance(
        intrinsics,
        output,
        moge_manifest,
        fix_principal_point=True,
    )
    sidecar = output.with_suffix(".generation.json")

    frame = intrinsics / "000001.json"
    original_frame = frame.read_bytes()
    frame.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MoGe generation validation failed"):
        stabilizer.validate_stable_intrinsics_manifest(sidecar)
    frame.write_bytes(original_frame)

    output.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output bytes"):
        stabilizer.validate_stable_intrinsics_manifest(sidecar)


def test_refuses_to_overwrite_either_member_of_a_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intrinsics, moge_manifest = _moge_generation(tmp_path, monkeypatch)
    output = tmp_path / "stable.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stabilizer.stabilize_intrinsics_with_provenance(
            intrinsics,
            output,
            moge_manifest,
        )


def test_legacy_api_remains_available_without_a_generation_manifest(
    tmp_path: Path,
) -> None:
    intrinsics = tmp_path / "intrinsics"
    intrinsics.mkdir()
    for index, focal_length in enumerate((900.0, 1000.0, 1100.0)):
        value = {
            "fx": focal_length,
            "fy": focal_length,
            "cx": 301.0 + index,
            "cy": 201.0 + index,
            "width": 640,
            "height": 480,
        }
        (intrinsics / f"{index:06d}.json").write_text(
            json.dumps(value),
            encoding="utf-8",
        )

    output = tmp_path / "legacy.json"
    stable = stabilizer.stabilize_intrinsics(
        str(intrinsics),
        str(output),
    )

    assert stable.fx == 1000.0
    assert stable.cx == 302.0
    assert json.loads(output.read_text()) == stable.to_dict()
