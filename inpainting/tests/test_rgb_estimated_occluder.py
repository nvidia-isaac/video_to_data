from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from inpainting.contracts import ContractError
from inpainting.occluder_depth import (
    OCCLUDER_METADATA_NAME,
    validate_occluder_depth_bundle,
)
from inpainting.rgb_estimated_occluder import (
    build_rgb_estimated_occluder,
    decode_v2d_inverse_depth,
    effective_e2fgvi_removal_mask,
    validate_rgb_only_moge_generation,
    validate_rgb_only_sam2_generation,
)
from inpainting.video_io import probe_video
from v2d.moge.docker import run_video_to_depth as moge_runner


SAM2_GENERATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "reconstruction"
    / "modules"
    / "v2d_sam2"
    / "lib"
    / "generation.py"
)
SAM2_GENERATION_SPEC = importlib.util.spec_from_file_location(
    "_rgb_occluder_sam2_generation_test", SAM2_GENERATION_PATH
)
assert SAM2_GENERATION_SPEC is not None and SAM2_GENERATION_SPEC.loader is not None
sam2_generation = importlib.util.module_from_spec(SAM2_GENERATION_SPEC)
SAM2_GENERATION_SPEC.loader.exec_module(sam2_generation)


def _write_video(path: Path, frame_count: int, size: tuple[int, int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((size[1], size[0], 3), 20 + index, dtype=np.uint8))
    writer.release()


def _encode_depth(depth: np.ndarray) -> np.ndarray:
    return np.clip(65535.0 / (depth + 1.0), 0, 65535).astype(np.uint16)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_moge_generation(
    *,
    video: Path,
    checkpoint: Path,
    depth_dir: Path,
    intrinsics_dir: Path,
    validity_dir: Path,
    image_id: str,
    input_intrinsics: Path | None = None,
) -> Path:
    requested_outputs = ["depth", "intrinsics", "mask"]
    snapshot = moge_runner._source_snapshot(
        video_path=video.resolve(),
        input_intrinsics_path=(
            None if input_intrinsics is None else input_intrinsics.resolve()
        ),
        checkpoint_path=checkpoint.resolve(),
        image_id=image_id,
        batch_size=8,
        requested_outputs=requested_outputs,
        dev=False,
    )
    static_identity = moge_runner._static_identity(snapshot)
    outputs = moge_runner._outputs_snapshot(
        {
            "depth": depth_dir.resolve(),
            "intrinsics": intrinsics_dir.resolve(),
            "mask": validity_dir.resolve(),
        }
    )
    frame_count = outputs["depth"]["file_count"]
    manifest = {
        "schema_version": moge_runner.RUN_GENERATION_SCHEMA,
        "state": "complete",
        **snapshot,
        "static_identity": static_identity,
        "expected_frames": {
            "count": frame_count,
            "indices": [0, frame_count - 1],
        },
        "outputs": outputs,
        "generation_id": moge_runner._generation_id(static_identity, outputs),
    }
    path = depth_dir / moge_runner.RUN_GENERATION_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _sam2_prompt_payload(
    *,
    video: Path,
    geometry,
    sequence_id: str,
    object_ids: tuple[int, ...],
) -> dict:
    prompts = []
    for index, object_id in enumerate(object_ids):
        x0 = min(float(index), float(geometry.width - 2))
        prompts.append(
            {
                "frame_index": 0,
                "object_id": object_id,
                "points": None,
                "point_labels": None,
                "box": {
                    "x0": x0,
                    "y0": 0.0,
                    "x1": x0 + 1.0,
                    "y1": 1.0,
                },
                "mask_path": None,
            }
        )
    return {
        "prompts": prompts,
        "metadata": {
            "schema_version": "v2d.inpainting.sam2-prompts/v1",
            "sequence_id": sequence_id,
            "source_video": str(video.resolve()),
            "geometry": geometry.as_dict(),
            "role": "rgb_only_tool_and_target_segmentation",
            "initialization": "human_box_prompts_on_rgb_frame_0",
            "object_ids": {
                str(object_id): f"object_{object_id}" for object_id in object_ids
            },
        },
    }


def _write_sam2_generation(
    *,
    video: Path,
    prompts: Path,
    checkpoint: Path,
    masks_dir: Path,
    image_id: str,
    object_ids: tuple[int, ...],
    frame_count: int,
) -> Path:
    prompt_payload = json.loads(prompts.read_text())
    staged_prompts = prompts.with_name(f".{prompts.stem}.container.json")
    staged_prompts.write_text(json.dumps(prompt_payload, indent=2))
    static_identity = sam2_generation.build_static_identity(
        str(video),
        str(staged_prompts),
        str(checkpoint.parent),
        prompt_payload["prompts"],
        image_id,
    )
    sam2_generation.commit_generation(
        masks_dir, static_identity, list(object_ids), frame_count
    )
    return masks_dir / sam2_generation.RUN_GENERATION_FILENAME


def _sam2_validation_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict]:
    video = tmp_path / "source.mp4"
    _write_video(video, 2, (8, 6))
    geometry = probe_video(video)
    checkpoint = tmp_path / "sam2.1_hiera_large.pt"
    checkpoint.write_bytes(b"sam2-checkpoint")
    monkeypatch.setattr(
        "inpainting.rgb_estimated_occluder.SAM2_CHECKPOINT_SHA256",
        _sha256(checkpoint),
    )
    prompts = tmp_path / "prompts.json"
    prompt_payload = _sam2_prompt_payload(
        video=video,
        geometry=geometry,
        sequence_id="sequence",
        object_ids=(1, 2),
    )
    prompts.write_text(json.dumps(prompt_payload))
    masks_dir = tmp_path / "masks"
    object_files: dict[int, list[Path]] = {}
    for object_id in (1, 2):
        object_dir = masks_dir / str(object_id)
        object_dir.mkdir(parents=True)
        object_files[object_id] = []
        for index in range(geometry.frame_count):
            path = object_dir / f"{index:06d}.png"
            assert cv2.imwrite(
                str(path), np.full((geometry.height, geometry.width), 255, np.uint8)
            )
            object_files[object_id].append(path)
    image_id = "sha256:" + "5" * 64
    manifest = _write_sam2_generation(
        video=video,
        prompts=prompts,
        checkpoint=checkpoint,
        masks_dir=masks_dir,
        image_id=image_id,
        object_ids=(1, 2),
        frame_count=geometry.frame_count,
    )
    kwargs = {
        "sequence_id": "sequence",
        "source_video": video.resolve(),
        "geometry": geometry,
        "prompts_path": prompts.resolve(),
        "object_files": object_files,
        "sam2_checkpoint": checkpoint.resolve(),
        "sam2_image_id": image_id,
    }
    resources = {
        "video": video,
        "prompts": prompts,
        "checkpoint": checkpoint,
        "masks_dir": masks_dir,
        "manifest": manifest,
        "prompt_payload": prompt_payload,
    }
    return kwargs, resources


def test_inverse_depth_decode_round_trips_quantized_metric_z(tmp_path: Path) -> None:
    path = tmp_path / "depth.png"
    source = np.array([[0.5, 1.0], [2.0, np.inf]], dtype=np.float32)
    assert cv2.imwrite(str(path), _encode_depth(source))
    decoded = decode_v2d_inverse_depth(path)
    np.testing.assert_allclose(decoded[:2, :1], source[:2, :1], atol=2e-4)
    assert np.isposinf(decoded[1, 1])


def test_effective_mask_matches_resize_cross_dilate_resize() -> None:
    source = np.zeros((6, 8), dtype=bool)
    source[2, 2] = True
    actual = effective_e2fgvi_removal_mask(
        source,
        processing_width=4,
        processing_height=3,
        dilation_kernel=3,
        dilation_iterations=1,
    )
    small = cv2.resize(source.astype(np.uint8), (4, 3), interpolation=cv2.INTER_NEAREST)
    expected = cv2.dilate(
        small,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1,
    )
    expected = cv2.resize(expected, (8, 6), interpolation=cv2.INTER_NEAREST) > 0
    np.testing.assert_array_equal(actual, expected)


def test_build_rgb_bundle_gates_depth_by_objects_validity_and_removal(
    tmp_path: Path, monkeypatch
) -> None:
    frame_count = 2
    size = (8, 6)
    video = tmp_path / "source.mp4"
    _write_video(video, frame_count, size)
    geometry = probe_video(video)
    assert geometry.frame_count == frame_count

    depth_dir = tmp_path / "depth"
    intrinsics_dir = tmp_path / "intrinsics"
    validity_dir = tmp_path / "validity"
    masks_dir = tmp_path / "sam2"
    for directory in (
        depth_dir,
        intrinsics_dir,
        validity_dir,
        masks_dir / "1",
        masks_dir / "2",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    depth_value = np.full((size[1], size[0]), 0.75, dtype=np.float32)
    object_one = np.zeros(depth_value.shape, dtype=np.uint8)
    object_one[0, 0] = 255
    object_one[1, 1] = 255
    object_two = np.zeros(depth_value.shape, dtype=np.uint8)
    object_two[2, 2] = 255
    validity = np.full(depth_value.shape, 255, dtype=np.uint8)
    validity[2, 2] = 0
    for index in range(frame_count):
        assert cv2.imwrite(
            str(depth_dir / f"{index:06d}.png"), _encode_depth(depth_value)
        )
        assert cv2.imwrite(str(validity_dir / f"{index:06d}.png"), validity)
        assert cv2.imwrite(str(masks_dir / "1" / f"{index:06d}.png"), object_one)
        assert cv2.imwrite(str(masks_dir / "2" / f"{index:06d}.png"), object_two)
        (intrinsics_dir / f"{index:06d}.json").write_text(
            json.dumps(
                {
                    "fx": 5.0,
                    "fy": 5.0,
                    "cx": 4.0,
                    "cy": 3.0,
                    "width": size[0],
                    "height": size[1],
                }
            )
        )
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps(
            _sam2_prompt_payload(
                video=video,
                geometry=geometry,
                sequence_id="test_sequence",
                object_ids=(1, 2),
            )
        )
    )
    arm_mask = tmp_path / "arm.npy"
    arms = np.zeros((frame_count, size[1], size[0]), dtype=bool)
    arms[:, 1, 1] = True
    np.save(arm_mask, arms)
    e2fgvi = tmp_path / "e2fgvi.json"
    e2fgvi.write_text(
        json.dumps(
            {
                "run": {
                    "source_video": geometry.as_dict(),
                    "processing_resolution": {
                        "width": size[0],
                        "height": size[1],
                    },
                    "parameters": {
                        "dilation_kernel": 3,
                        "dilation_iterations": 0,
                    },
                }
            }
        )
    )
    moge_checkpoint = tmp_path / "moge.pt"
    sam2_checkpoint = tmp_path / "sam2.1_hiera_large.pt"
    moge_checkpoint.write_bytes(b"moge-test")
    sam2_checkpoint.write_bytes(b"sam2-test")
    monkeypatch.setattr(
        "inpainting.rgb_estimated_occluder.MOGE_MODEL_SHA256",
        _sha256(moge_checkpoint),
    )
    monkeypatch.setattr(
        "inpainting.rgb_estimated_occluder.SAM2_CHECKPOINT_SHA256",
        _sha256(sam2_checkpoint),
    )
    moge_image_id = "sha256:" + "1" * 64
    moge_generation = _write_moge_generation(
        video=video,
        checkpoint=moge_checkpoint,
        depth_dir=depth_dir,
        intrinsics_dir=intrinsics_dir,
        validity_dir=validity_dir,
        image_id=moge_image_id,
    )
    sam2_image_id = "sha256:" + "2" * 64
    _write_sam2_generation(
        video=video,
        prompts=prompts,
        checkpoint=sam2_checkpoint,
        masks_dir=masks_dir,
        image_id=sam2_image_id,
        object_ids=(1, 2),
        frame_count=frame_count,
    )

    output = tmp_path / "bundle"
    metadata = build_rgb_estimated_occluder(
        sequence_id="test_sequence",
        source_video=video,
        moge_depth_dir=depth_dir,
        moge_intrinsics_dir=intrinsics_dir,
        moge_validity_dir=validity_dir,
        moge_generation=moge_generation,
        sam2_masks_dir=masks_dir,
        sam2_object_ids=(1, 2),
        sam2_prompts=prompts,
        arm_mask=arm_mask,
        e2fgvi_metadata=e2fgvi,
        moge_checkpoint=moge_checkpoint,
        sam2_checkpoint=sam2_checkpoint,
        output_dir=output,
        moge_image_id=moge_image_id,
        sam2_image_id=sam2_image_id,
    )

    assert metadata["source_modalities"] == ["rgb"]
    assert metadata["estimation"]["uses_ground_truth"] is False
    assert metadata["estimation"]["moge"]["input_intrinsics"] is None
    mask = np.load(output / "occluder_mask.npy")
    expected = np.zeros_like(mask)
    expected[:, 0, 0] = True
    np.testing.assert_array_equal(mask, expected)
    depth = np.load(output / "occluder_depth.npy")
    assert np.isfinite(depth[mask]).all()
    assert np.isposinf(depth[~mask]).all()
    validated, _ = validate_occluder_depth_bundle(
        output / OCCLUDER_METADATA_NAME, geometry
    )
    assert validated == metadata


def test_rgb_only_sam2_validator_accepts_byte_exact_human_box_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, resources = _sam2_validation_case(tmp_path, monkeypatch)

    manifest = validate_rgb_only_sam2_generation(resources["manifest"], **kwargs)

    assert manifest["state"] == "complete"
    assert manifest["expected"] == {"object_ids": [1, 2], "frame_count": 2}


def test_rgb_only_sam2_validator_rejects_non_box_prompt_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, resources = _sam2_validation_case(tmp_path, monkeypatch)
    payload = resources["prompt_payload"]
    payload["prompts"][0]["box"] = None
    payload["prompts"][0]["points"] = [{"x": 1.0, "y": 1.0}]
    payload["prompts"][0]["point_labels"] = [1]
    resources["prompts"].write_text(json.dumps(payload))

    with pytest.raises(ContractError, match="permits boxes only"):
        validate_rgb_only_sam2_generation(resources["manifest"], **kwargs)


def test_rgb_only_sam2_validator_rejects_gt_declared_prompt_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, resources = _sam2_validation_case(tmp_path, monkeypatch)
    payload = resources["prompt_payload"]
    payload["metadata"]["initialization"] = "ground_truth_object_mask"
    resources["prompts"].write_text(json.dumps(payload))

    with pytest.raises(ContractError, match="human RGB box prompts"):
        validate_rgb_only_sam2_generation(resources["manifest"], **kwargs)


def test_rgb_only_sam2_validator_rejects_prompt_changed_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, resources = _sam2_validation_case(tmp_path, monkeypatch)
    payload = resources["prompt_payload"]
    payload["prompts"][0]["box"]["x1"] = 2.0
    resources["prompts"].write_text(json.dumps(payload))

    with pytest.raises(ContractError, match="pinned video, prompts"):
        validate_rgb_only_sam2_generation(resources["manifest"], **kwargs)


def test_rgb_only_sam2_validator_rejects_mask_changed_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, resources = _sam2_validation_case(tmp_path, monkeypatch)
    mask = resources["masks_dir"] / "1" / "000001.png"
    assert cv2.imwrite(str(mask), np.zeros((6, 8), np.uint8))

    with pytest.raises(ContractError, match="mask bytes"):
        validate_rgb_only_sam2_generation(resources["manifest"], **kwargs)


def test_rgb_only_sam2_validator_rejects_unpinned_source_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, resources = _sam2_validation_case(tmp_path, monkeypatch)
    manifest = json.loads(resources["manifest"].read_text())
    manifest["static_identity"]["implementation_sources"]["sam2_utils.py"]["sha256"] = (
        "0" * 64
    )
    resources["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="source commit"):
        validate_rgb_only_sam2_generation(resources["manifest"], **kwargs)


def test_rgb_only_validator_rejects_a_valid_generation_with_intrinsics_prior(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    known_k = tmp_path / "known_intrinsics.json"
    known_k.write_text('{"fx": 1000}')
    directories = {
        name: tmp_path / name for name in ("depth", "intrinsics", "validity")
    }
    for directory in directories.values():
        directory.mkdir()
    (directories["depth"] / "000000.png").write_bytes(b"depth")
    (directories["intrinsics"] / "000000.json").write_bytes(b"intrinsics")
    (directories["validity"] / "000000.png").write_bytes(b"validity")
    image_id = "sha256:" + "3" * 64
    manifest = _write_moge_generation(
        video=video,
        checkpoint=checkpoint,
        depth_dir=directories["depth"],
        intrinsics_dir=directories["intrinsics"],
        validity_dir=directories["validity"],
        image_id=image_id,
        input_intrinsics=known_k,
    )

    with pytest.raises(ContractError, match="no input intrinsics"):
        validate_rgb_only_moge_generation(
            manifest,
            source_video=video,
            depth_files=[directories["depth"] / "000000.png"],
            intrinsics_files=[directories["intrinsics"] / "000000.json"],
            validity_files=[directories["validity"] / "000000.png"],
            moge_checkpoint=checkpoint,
            moge_image_id=image_id,
        )


def test_rgb_only_validator_rejects_output_changed_after_commit(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    directories = {
        name: tmp_path / name for name in ("depth", "intrinsics", "validity")
    }
    for directory in directories.values():
        directory.mkdir()
    depth = directories["depth"] / "000000.png"
    intrinsics = directories["intrinsics"] / "000000.json"
    validity = directories["validity"] / "000000.png"
    depth.write_bytes(b"depth")
    intrinsics.write_bytes(b"intrinsics")
    validity.write_bytes(b"validity")
    image_id = "sha256:" + "4" * 64
    manifest = _write_moge_generation(
        video=video,
        checkpoint=checkpoint,
        depth_dir=directories["depth"],
        intrinsics_dir=directories["intrinsics"],
        validity_dir=directories["validity"],
        image_id=image_id,
    )
    depth.write_bytes(b"tampered")

    with pytest.raises(ContractError, match="committed generation"):
        validate_rgb_only_moge_generation(
            manifest,
            source_video=video,
            depth_files=[depth],
            intrinsics_files=[intrinsics],
            validity_files=[validity],
            moge_checkpoint=checkpoint,
            moge_image_id=image_id,
        )
