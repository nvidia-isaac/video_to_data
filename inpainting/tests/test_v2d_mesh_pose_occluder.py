from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from inpainting.contracts import ContractError, VideoGeometry
from inpainting.occluder_depth import (
    OCCLUDER_METADATA_NAME,
    validate_occluder_depth_bundle,
)
from inpainting import v2d_mesh_pose_occluder as module
from inpainting.v2d_mesh_pose_occluder import (
    load_camera_model,
    load_object_to_camera_pose,
    load_v2d_mesh_pose_inputs,
    metric_depth_layer,
    nearest_depth_union,
    opencv_pose_to_pyrender_pose,
    render_v2d_mesh_pose_occluder,
)


GEOMETRY = VideoGeometry(frame_count=2, width=8, height=8, fps=30.0)
SEQUENCE_ID = "taco_cut__knife__plate_20231013_105"
IMAGE_ID = "sha256:" + "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _generation_record(path: Path) -> dict[str, object]:
    record = _record(path)
    return {"size_bytes": record["bytes"], "sha256": record["sha256"]}


def _canonical_json_record(value: object) -> dict[str, object]:
    payload = json.dumps(value, indent=2).encode()
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _weight_records(
    identities: dict[str, tuple[str, int, str]], *, root: str
) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "record": {
                "path": f"/{root}/{name}/{basename}",
                "bytes": byte_count,
                "sha256": digest,
            },
        }
        for name, (basename, byte_count, digest) in sorted(identities.items())
    ]


def _write_intrinsics(path: Path, **overrides: object) -> None:
    value: dict[str, object] = {
        "fx": 6.0,
        "fy": 6.0,
        "cx": 4.0,
        "cy": 4.0,
        "width": 8,
        "height": 8,
    }
    value.update(overrides)
    _write_json(path, value)


def _foundation_pose(*, translation: tuple[float, float, float]) -> dict[str, object]:
    return {
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": list(translation),
        "scale": [1.0, 1.0, 1.0],
    }


def _write_track(root: Path, name: str) -> tuple[Path, Path]:
    object_root = root / name
    object_root.mkdir()
    mesh = object_root / "metric_mesh.obj"
    mesh.write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n", encoding="utf-8")
    (object_root / "sam3d_mesh.glb").write_bytes(b"fake SAM3D mesh")
    _write_json(
        object_root / "sam3d_transform.json",
        {
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.7],
            "scale": [0.2, 0.2, 0.2],
        },
    )
    (object_root / "mesh_pretransformed.glb").write_bytes(b"fake transformed mesh")
    _write_json(object_root / "scale.json", {"scale": 1.25})
    _write_json(
        object_root / "scale_registration_pose.json",
        _foundation_pose(translation=(0.0, 0.0, 0.7)),
    )
    raw_poses = object_root / "poses_raw"
    raw_poses.mkdir()
    poses = object_root / "poses_smoothed"
    poses.mkdir()
    for index in range(GEOMETRY.frame_count):
        pose = _foundation_pose(translation=(0.01 * index, 0.0, 0.7))
        _write_json(raw_poses / f"{index:06d}.json", pose)
        _write_json(poses / f"{index:06d}.json", pose)
    return mesh, poses


def _write_rgb_evidence(
    root: Path,
    *,
    source_video: Path,
    legacy_intrinsics: Path,
    names: tuple[str, ...],
) -> tuple[Path, Path, Path, Path, Path, dict[str, int]]:
    moge_root = root / "moge"
    intrinsics_dir = moge_root / "intrinsics"
    intrinsics_dir.mkdir(parents=True)
    intrinsics_paths = []
    for index in range(GEOMETRY.frame_count):
        path = intrinsics_dir / f"{index:06d}.json"
        _write_intrinsics(path)
        intrinsics_paths.append(path)
    moge_generation = moge_root / "run_generation.json"
    _write_json(
        moge_generation,
        {
            "schema_version": module.MOGE_GENERATION_SCHEMA,
            "state": "complete",
            "generation_id": "sha256:" + "b" * 64,
            "parameters": {
                "input_intrinsics_path": None,
                "intrinsics_mode": "estimated_from_rgb",
                "frame_index_origin": 0,
                "requested_outputs": ["depth", "intrinsics", "mask", "points"],
            },
            "sources": {
                "input_intrinsics": None,
                "video": {
                    "path": str(source_video.resolve()),
                    **_generation_record(source_video),
                },
            },
            "source_revisions": {
                "moge_repository": module.MOGE_REPOSITORY,
                "moge_huggingface_revision": module.MOGE_REVISION,
                "moge_source_commit": module.MOGE_SOURCE_COMMIT,
            },
            "model": {
                "checkpoint": {
                    "path": "/weights/moge/model.pt",
                    "size_bytes": module.MOGE_MODEL_BYTES,
                    "sha256": module.MOGE_MODEL_SHA256,
                }
            },
            "execution_environment": {"container_image_id": IMAGE_ID},
            "expected_frames": {
                "count": GEOMETRY.frame_count,
                "indices": [0, GEOMETRY.frame_count - 1],
            },
            "outputs": {
                "intrinsics": {
                    "files": {
                        path.name: _generation_record(path) for path in intrinsics_paths
                    }
                }
            },
        },
    )

    stable_output = root / "intrinsics_stable_rgb_only.json"
    stable_output.write_text(legacy_intrinsics.read_text(), encoding="utf-8")
    stable_parameters = {
        "algorithm": "coordinate_wise_temporal_median/v1",
        "dimension_policy": "require_constant_across_frames",
        "fix_principal_point": True,
        "frame_order": "contiguous_zero_based_six_digit_filenames",
        "principal_point_policy": "image_center",
    }
    source_intrinsics = {
        "files": {path.name: _generation_record(path) for path in intrinsics_paths}
    }
    stable_reproduction = root / "intrinsics_stable_rgb_only.generation.json"
    _write_json(
        stable_reproduction,
        {
            "schema_version": module.STABLE_INTRINSICS_GENERATION_SCHEMA,
            "state": "complete",
            "generation_id": "sha256:" + "c" * 64,
            "parameters": stable_parameters,
            "implementation_sources": {
                module.STABLE_INTRINSICS_IMPLEMENTATION_PATH: {
                    "path": module.STABLE_INTRINSICS_IMPLEMENTATION_PATH,
                    "size_bytes": module.STABLE_INTRINSICS_IMPLEMENTATION_BYTES,
                    "sha256": module.STABLE_INTRINSICS_IMPLEMENTATION_SHA256,
                }
            },
            "sources": {
                "intrinsics": source_intrinsics,
                "moge_generation_id": "sha256:" + "b" * 64,
                "moge_generation_manifest": {
                    "path": str(moge_generation.resolve()),
                    **_generation_record(moge_generation),
                },
                "moge_schema_version": module.MOGE_GENERATION_SCHEMA,
                "parameters": stable_parameters,
            },
            "static_identity": {
                "implementation_sources": {
                    module.STABLE_INTRINSICS_IMPLEMENTATION_PATH: {
                        "size_bytes": module.STABLE_INTRINSICS_IMPLEMENTATION_BYTES,
                        "sha256": module.STABLE_INTRINSICS_IMPLEMENTATION_SHA256,
                    }
                },
                "intrinsics": source_intrinsics,
                "moge_generation_id": "sha256:" + "b" * 64,
                "moge_generation_manifest": _generation_record(moge_generation),
                "moge_schema_version": module.MOGE_GENERATION_SCHEMA,
                "parameters": stable_parameters,
            },
            "output": {
                "stable_intrinsics": {
                    "path": str(stable_output.resolve()),
                    **_generation_record(stable_output),
                },
                "values": {
                    "fx": 6.0,
                    "fy": 6.0,
                    "cx": 4.0,
                    "cy": 4.0,
                    "width": 8,
                    "height": 8,
                },
            },
        },
    )

    object_ids = {name: index + 1 for index, name in enumerate(names)}
    sam2_root = root / "sam2_masks"
    sam2_root.mkdir()
    prompts = root / "sam2_prompts.json"
    prompt_payload = {
        "prompts": [
            {
                "frame_index": 0,
                "object_id": object_ids[name],
                "points": None,
                "point_labels": None,
                "box": {
                    "x0": float(index),
                    "y0": float(index),
                    "x1": float(index + 2),
                    "y1": float(index + 2),
                },
                "mask_path": None,
            }
            for index, name in enumerate(names)
        ],
        "metadata": {
            "schema_version": module.SAM2_PROMPTS_SCHEMA,
            "sequence_id": SEQUENCE_ID,
            "source_video": str(source_video.resolve()),
            "geometry": GEOMETRY.as_dict(),
            "role": "rgb_only_tool_and_target_segmentation",
            "initialization": "human_box_prompts_on_rgb_frame_0",
            "object_ids": {str(object_ids[name]): name for name in names},
        },
    }
    _write_json(prompts, prompt_payload)
    sam2_outputs: dict[str, object] = {}
    for name in names:
        object_id = object_ids[name]
        mask_dir = sam2_root / str(object_id)
        mask_dir.mkdir()
        files = {}
        for frame_index in range(GEOMETRY.frame_count):
            mask = mask_dir / f"{frame_index:06d}.png"
            mask.write_bytes(f"mask {name} {frame_index}".encode())
            files[mask.name] = _generation_record(mask)
        sam2_outputs[str(object_id)] = {"files": files}
    sam2_generation = sam2_root / "run_generation.json"
    _write_json(
        sam2_generation,
        {
            "schema_version": module.SAM2_GENERATION_SCHEMA,
            "state": "complete",
            "expected": {
                "frame_count": GEOMETRY.frame_count,
                "object_ids": sorted(object_ids.values()),
            },
            "static_identity": {
                "checkpoint": {
                    "artifact": {
                        "size_bytes": module.SAM2_CHECKPOINT_BYTES,
                        "sha256": module.SAM2_CHECKPOINT_SHA256,
                    }
                },
                "execution_environment": {"container_image_id": IMAGE_ID},
                "prompts_json": _canonical_json_record(prompt_payload),
                "video": {
                    "kind": "file",
                    "artifact": _generation_record(source_video),
                },
            },
            "outputs": {"objects": sam2_outputs},
        },
    )
    return (
        moge_generation,
        intrinsics_dir,
        stable_reproduction,
        sam2_generation,
        prompts,
        object_ids,
    )


def _write_lineage(
    path: Path,
    *,
    source_video: Path,
    intrinsics: Path,
    tracks: dict[str, tuple[Path, Path]],
) -> Path:
    names = tuple(sorted(tracks))
    parameters = module.canonical_object_lineage_parameters(names)
    (
        moge_generation,
        moge_intrinsics_dir,
        stable_intrinsics_reproduction,
        sam2_generation,
        sam2_prompts,
        object_ids,
    ) = _write_rgb_evidence(
        path.parent,
        source_video=source_video,
        legacy_intrinsics=intrinsics,
        names=names,
    )
    manifest = {
        "schema_version": module.UPSTREAM_LINEAGE_SCHEMA,
        "state": "complete",
        "run_id": "11111111-1111-4111-8111-111111111111",
        "sequence_id": SEQUENCE_ID,
        "geometry": GEOMETRY.as_dict(),
        "source_video": _record(source_video),
        "rgb_evidence": {
            "moge_generation": _record(moge_generation),
            "moge_intrinsics": [
                _record(moge_intrinsics_dir / f"{index:06d}.json")
                for index in range(GEOMETRY.frame_count)
            ],
            "stable_intrinsics_reproduction": _record(stable_intrinsics_reproduction),
            "sam2_generation": _record(sam2_generation),
            "sam2_prompts": _record(sam2_prompts),
        },
        "artifacts": {
            "intrinsics": _record(intrinsics),
            "objects": [
                {
                    "name": name,
                    "sam2_object_id": object_ids[name],
                    "mesh": _record(tracks[name][0]),
                    "poses": [
                        _record(tracks[name][1] / f"{index:06d}.json")
                        for index in range(GEOMETRY.frame_count)
                    ],
                    "chain": {
                        "sam3d_mesh": _record(
                            tracks[name][0].parent / "sam3d_mesh.glb"
                        ),
                        "sam3d_transform": _record(
                            tracks[name][0].parent / "sam3d_transform.json"
                        ),
                        "mesh_pretransformed": _record(
                            tracks[name][0].parent / "mesh_pretransformed.glb"
                        ),
                        "scale": _record(tracks[name][0].parent / "scale.json"),
                        "scale_registration_pose": _record(
                            tracks[name][0].parent / "scale_registration_pose.json"
                        ),
                        "raw_poses": [
                            _record(
                                tracks[name][0].parent
                                / "poses_raw"
                                / f"{index:06d}.json"
                            )
                            for index in range(GEOMETRY.frame_count)
                        ],
                    },
                }
                for name in names
            ],
        },
        "stages": {
            "sam3d": {
                "model": {
                    "repository": module.SAM3D_REPOSITORY,
                    "revision": module.SAM3D_REVISION,
                },
                "container": {"image": "v2d_sam3d:test", "image_id": IMAGE_ID},
                "weights": _weight_records(
                    module.SAM3D_WEIGHT_IDENTITIES, root="sam3d"
                ),
                "parameters": parameters["sam3d"],
            },
            "foundation_pose": {
                "model": {
                    "repository": module.FOUNDATIONPOSE_REPOSITORY,
                    "revision": module.FOUNDATIONPOSE_REVISION,
                },
                "container": {
                    "image": "v2d_foundation_pose:test",
                    "image_id": IMAGE_ID,
                },
                "weights": _weight_records(
                    module.FOUNDATIONPOSE_WEIGHT_IDENTITIES,
                    root="foundation_pose",
                ),
                "parameters": parameters["foundation_pose"],
            },
        },
    }
    _write_json(path, manifest)
    return path


def _load_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> module.V2DMeshPoseInputs:
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"synthetic source video")
    intrinsics = tmp_path / "intrinsics.json"
    _write_intrinsics(intrinsics)
    knife_mesh, knife_poses = _write_track(tmp_path, "knife")
    board_mesh, board_poses = _write_track(tmp_path, "cutting_board")
    lineage = _write_lineage(
        tmp_path / "upstream_lineage.json",
        source_video=source_video,
        intrinsics=intrinsics,
        tracks={
            "knife": (knife_mesh, knife_poses),
            "cutting_board": (board_mesh, board_poses),
        },
    )
    monkeypatch.setattr(module, "probe_video", lambda _: GEOMETRY)
    # Supply reverse lexical order to prove that input ordering cannot alter
    # layer order or metadata.
    return load_v2d_mesh_pose_inputs(
        sequence_id=SEQUENCE_ID,
        source_video=source_video,
        intrinsics_path=intrinsics,
        object_specs=(
            ("knife", knife_mesh, knife_poses),
            ("cutting_board", board_mesh, board_poses),
        ),
        lineage_manifest=lineage,
    )


def test_cv_to_gl_is_a_left_camera_basis_change() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    point_object = np.array([0.2, -0.1, 0.4, 1.0])

    pose_gl = opencv_pose_to_pyrender_pose(pose)

    np.testing.assert_allclose(
        pose_gl @ point_object, module.CV_TO_OPENGL @ pose @ point_object
    )
    np.testing.assert_allclose(pose_gl[:3, 3], [1.0, -2.0, -3.0])
    # Right-multiplying or conjugating would incorrectly reinterpret the mesh's
    # object coordinate basis.
    assert not np.allclose(pose_gl, module.CV_TO_OPENGL @ pose @ module.CV_TO_OPENGL)


def test_metric_layers_and_nearest_union_use_positive_finite_camera_z() -> None:
    far = np.array([[0.0, 0.8], [np.nan, 0.9]], dtype=np.float64)
    near = np.array([[0.4, 0.6], [np.inf, -1.0]], dtype=np.float32)

    normalized = metric_depth_layer(far, (2, 2))
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(
        np.isfinite(normalized), [[False, True], [False, True]]
    )

    mask, depth = nearest_depth_union((far, near))
    np.testing.assert_array_equal(mask, [[True, True], [False, True]])
    np.testing.assert_allclose(depth[mask], [0.4, 0.6, 0.9])
    assert np.isposinf(depth[1, 0])


def test_nearest_union_rejects_empty_or_wrong_shape_layers() -> None:
    with pytest.raises(ContractError, match="At least one"):
        nearest_depth_union(())
    with pytest.raises(module.V2DMeshPoseOccluderError, match="does not match"):
        nearest_depth_union((np.ones((2, 2)), np.ones((3, 2))))


def test_foundationpose_and_matrix_pose_files_load_as_the_same_se3(
    tmp_path: Path,
) -> None:
    transform_path = tmp_path / "transform.json"
    matrix_path = tmp_path / "matrix.json"
    transform = _foundation_pose(translation=(0.1, -0.2, 0.7))
    expected = np.eye(4)
    expected[:3, 3] = [0.1, -0.2, 0.7]
    _write_json(transform_path, transform)
    _write_json(matrix_path, expected.tolist())

    from_transform, transform_format = load_object_to_camera_pose(transform_path)
    from_matrix, matrix_format = load_object_to_camera_pose(matrix_path)

    np.testing.assert_allclose(from_transform, expected)
    np.testing.assert_allclose(from_matrix, expected)
    assert transform_format == "foundationpose_transform3d_wxyz"
    assert matrix_format == "matrix_4x4"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(scale=[1.2, 1.2, 1.2]), "unit scale"),
        (lambda value: value.update(rotation=[0.0, 0.0, 0.0, 0.0]), "unit length"),
        (lambda value: value.update(translation=[0.0, float("nan"), 0.7]), "finite"),
    ],
)
def test_foundationpose_pose_contract_rejects_non_se3(
    tmp_path: Path, mutate, match: str
) -> None:
    pose = _foundation_pose(translation=(0.0, 0.0, 0.7))
    mutate(pose)
    path = tmp_path / "bad.json"
    _write_json(path, pose)
    with pytest.raises(ContractError, match=match):
        load_object_to_camera_pose(path)


def test_matrix_pose_rejects_reflection(tmp_path: Path) -> None:
    reflected = np.eye(4)
    reflected[0, 0] = -1.0
    path = tmp_path / "reflection.json"
    _write_json(path, reflected.tolist())
    with pytest.raises(ContractError, match="reflected"):
        load_object_to_camera_pose(path)


def test_camera_model_is_exact_and_matches_target_resolution(tmp_path: Path) -> None:
    path = tmp_path / "intrinsics.json"
    _write_intrinsics(path)
    camera = load_camera_model(path, expected_width=8, expected_height=8)
    np.testing.assert_allclose(
        camera.matrix, [[6.0, 0.0, 4.0], [0.0, 6.0, 4.0], [0.0, 0.0, 1.0]]
    )

    _write_intrinsics(path, width=9)
    with pytest.raises(ContractError, match="does not match target"):
        load_camera_model(path, expected_width=8, expected_height=8)

    _write_intrinsics(path, model="pinhole")
    with pytest.raises(ContractError, match="exactly"):
        load_camera_model(path, expected_width=8, expected_height=8)


def test_input_loader_sorts_objects_and_requires_complete_pose_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    assert [track.name for track in inputs.objects] == ["cutting_board", "knife"]
    assert all(track.object_to_camera.shape == (2, 4, 4) for track in inputs.objects)
    assert len(inputs.provenance_inputs) == 3 + 2 * (1 + GEOMETRY.frame_count)

    (inputs.objects[0].poses_dir / "notes.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="exactly 2 numbered"):
        load_v2d_mesh_pose_inputs(
            sequence_id=SEQUENCE_ID,
            source_video=inputs.source_video,
            intrinsics_path=inputs.camera.path,
            object_specs=(
                (
                    "cutting_board",
                    inputs.objects[0].mesh_path,
                    inputs.objects[0].poses_dir,
                ),
            ),
            lineage_manifest=inputs.upstream_lineage.path,
        )


def test_input_loader_rejects_duplicate_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    intrinsics = tmp_path / "intrinsics.json"
    _write_intrinsics(intrinsics)
    mesh, poses = _write_track(tmp_path, "knife")
    other_mesh = tmp_path / "other.obj"
    other_mesh.write_text(mesh.read_text(), encoding="utf-8")
    monkeypatch.setattr(module, "probe_video", lambda _: GEOMETRY)
    with pytest.raises(ContractError, match="names must be unique"):
        load_v2d_mesh_pose_inputs(
            sequence_id=SEQUENCE_ID,
            source_video=source,
            intrinsics_path=intrinsics,
            object_specs=(("knife", mesh, poses), ("knife", other_mesh, poses)),
            lineage_manifest=tmp_path / "not_reached.json",
        )


class _FakeDepthRenderer:
    def __init__(self) -> None:
        self.closed = False

    def render(self, object_index: int, object_to_camera: np.ndarray) -> np.ndarray:
        assert object_to_camera.shape == (4, 4)
        depth = np.zeros((8, 8), dtype=np.float32)
        if object_index == 0:
            depth[1:5, 1:5] = 0.8
        else:
            depth[3:7, 3:7] = 0.4
        return depth

    def close(self) -> None:
        self.closed = True


def test_fake_renderer_commits_a_provenance_strict_nearest_union_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    fake = _FakeDepthRenderer()
    output = tmp_path / "occluder"

    metadata = render_v2d_mesh_pose_occluder(
        inputs,
        output,
        renderer_image="v2d_foundation_pose:test",
        renderer_image_id=IMAGE_ID,
        renderer_factory=lambda _inputs, _platform: fake,
    )

    assert fake.closed
    assert metadata["producer"]["name"] == "v2d_estimated_object"
    assert metadata["estimation"]["uses_ground_truth"] is False
    lineage = metadata["estimation"]["upstream_lineage"]
    assert lineage["manifest"]["sha256"] == inputs.upstream_lineage.record["sha256"]
    assert lineage["stages"]["sam3d"]["model"]["revision"] == module.SAM3D_REVISION
    assert metadata["statistics"]["total_occluder_pixels"] == 56
    validated, artifacts = validate_occluder_depth_bundle(
        output / OCCLUDER_METADATA_NAME, GEOMETRY
    )
    assert validated == metadata
    mask = np.load(artifacts["mask"])
    depth = np.load(artifacts["depth"])
    assert mask.dtype == np.bool_
    assert depth.dtype == np.float32
    assert mask.sum(axis=(1, 2)).tolist() == [28, 28]
    assert depth[0, 3, 3] == pytest.approx(0.4)
    assert depth[0, 1, 1] == pytest.approx(0.8)
    assert np.isposinf(depth[0, 0, 0])


def test_render_refuses_inputs_changed_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    inputs.objects[0].pose_paths[0].write_text("[]", encoding="utf-8")
    with pytest.raises(ContractError, match="does not fingerprint"):
        render_v2d_mesh_pose_occluder(
            inputs,
            tmp_path / "output",
            renderer_image="v2d_foundation_pose:test",
            renderer_image_id=IMAGE_ID,
            renderer_factory=lambda _inputs, _platform: _FakeDepthRenderer(),
        )


def _reload_inputs(inputs: module.V2DMeshPoseInputs) -> module.V2DMeshPoseInputs:
    return load_v2d_mesh_pose_inputs(
        sequence_id=inputs.sequence_id,
        source_video=inputs.source_video,
        intrinsics_path=inputs.camera.path,
        object_specs=tuple(
            (track.name, track.mesh_path, track.poses_dir) for track in inputs.objects
        ),
        lineage_manifest=inputs.upstream_lineage.path,
    )


def test_lineage_refuses_external_calibration_or_unbound_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    path = inputs.upstream_lineage.path
    manifest = json.loads(path.read_text())
    moge_generation_path = Path(manifest["rgb_evidence"]["moge_generation"]["path"])
    original_generation = json.loads(moge_generation_path.read_text())
    generation = json.loads(json.dumps(original_generation))
    generation["parameters"]["input_intrinsics_path"] = "/ground_truth/K.json"
    _write_json(moge_generation_path, generation)
    manifest["rgb_evidence"]["moge_generation"] = _record(moge_generation_path)
    _write_json(path, manifest)
    with pytest.raises(ContractError, match="no input calibration"):
        _reload_inputs(inputs)

    _write_json(moge_generation_path, original_generation)
    manifest["rgb_evidence"]["moge_generation"] = _record(moge_generation_path)
    manifest["artifacts"]["objects"][0]["mesh"]["sha256"] = "0" * 64
    _write_json(path, manifest)
    with pytest.raises(ContractError, match="does not fingerprint"):
        _reload_inputs(inputs)


def test_lineage_requires_rgb_stable_k_reproduction_to_match_consumed_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    lineage_path = inputs.upstream_lineage.path
    manifest = json.loads(lineage_path.read_text())
    reproduction_path = Path(
        manifest["rgb_evidence"]["stable_intrinsics_reproduction"]["path"]
    )
    reproduction = json.loads(reproduction_path.read_text())
    reproduced_k = Path(reproduction["output"]["stable_intrinsics"]["path"])
    _write_intrinsics(reproduced_k, fx=7.0)
    reproduction["output"]["stable_intrinsics"] = {
        "path": str(reproduced_k.resolve()),
        **_generation_record(reproduced_k),
    }
    reproduction["output"]["values"]["fx"] = 7.0
    _write_json(reproduction_path, reproduction)
    manifest["rgb_evidence"]["stable_intrinsics_reproduction"] = _record(
        reproduction_path
    )
    _write_json(lineage_path, manifest)

    with pytest.raises(ContractError, match="differs from the legacy K"):
        _reload_inputs(inputs)


def test_lineage_refuses_unpinned_models_containers_weights_and_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    path = inputs.upstream_lineage.path
    original = json.loads(path.read_text())

    manifest = json.loads(json.dumps(original))
    manifest["stages"]["sam3d"]["model"]["revision"] = "0" * 40
    _write_json(path, manifest)
    with pytest.raises(ContractError, match="model identity"):
        _reload_inputs(inputs)

    manifest = json.loads(json.dumps(original))
    manifest["stages"]["foundation_pose"]["container"]["image_id"] = "latest"
    _write_json(path, manifest)
    with pytest.raises(ContractError, match="immutable sha256"):
        _reload_inputs(inputs)

    manifest = json.loads(json.dumps(original))
    manifest["stages"]["sam3d"]["weights"][0]["record"]["sha256"] = "0" * 64
    _write_json(path, manifest)
    with pytest.raises(ContractError, match="identity does not match"):
        _reload_inputs(inputs)

    manifest = json.loads(json.dumps(original))
    del manifest["stages"]["sam3d"]["parameters"]["knife"]["seed"]
    _write_json(path, manifest)
    with pytest.raises(ContractError, match="must contain exactly"):
        _reload_inputs(inputs)


def test_render_refuses_lineage_manifest_changed_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    # Same semantics, different immutable manifest bytes.
    inputs.upstream_lineage.path.write_text(
        json.dumps(inputs.upstream_lineage.manifest, indent=4) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="changed after input validation"):
        render_v2d_mesh_pose_occluder(
            inputs,
            tmp_path / "output",
            renderer_image="v2d_foundation_pose:test",
            renderer_image_id=IMAGE_ID,
            renderer_factory=lambda _inputs, _platform: _FakeDepthRenderer(),
        )


def test_manifest_authoring_helper_writes_a_renderer_accepted_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)

    def fake_weight_records(_paths, expected, *, label):
        return _weight_records(expected, root=label.lower())

    monkeypatch.setattr(module, "_verified_weight_records", fake_weight_records)
    destination = tmp_path / "authored_lineage.json"
    prior = inputs.upstream_lineage.manifest
    manifest = module.write_upstream_object_lineage_manifest(
        destination,
        sequence_id=inputs.sequence_id,
        source_video=inputs.source_video,
        intrinsics_path=inputs.camera.path,
        object_specs=tuple(
            (track.name, track.mesh_path, track.poses_dir) for track in inputs.objects
        ),
        moge_generation=prior["rgb_evidence"]["moge_generation"]["path"],
        moge_intrinsics_dir=Path(
            prior["rgb_evidence"]["moge_intrinsics"][0]["path"]
        ).parent,
        stable_intrinsics_reproduction=prior["rgb_evidence"][
            "stable_intrinsics_reproduction"
        ]["path"],
        sam2_generation=prior["rgb_evidence"]["sam2_generation"]["path"],
        sam2_prompts=prior["rgb_evidence"]["sam2_prompts"]["path"],
        sam2_object_ids={
            item["name"]: item["sam2_object_id"]
            for item in prior["artifacts"]["objects"]
        },
        sam3d_weights_dir=tmp_path / "sam3d_weights",
        foundationpose_weights_dir=tmp_path / "foundationpose_weights",
        sam3d_container_image="v2d_sam3d:latest",
        sam3d_container_image_id=IMAGE_ID,
        foundationpose_container_image="v2d_foundation_pose:latest",
        foundationpose_container_image_id=IMAGE_ID,
        parameters=module.canonical_object_lineage_parameters(
            [track.name for track in inputs.objects]
        ),
        run_id="22222222-2222-4222-8222-222222222222",
    )
    assert destination.is_file()
    assert "source_policy" not in manifest
    accepted = load_v2d_mesh_pose_inputs(
        sequence_id=inputs.sequence_id,
        source_video=inputs.source_video,
        intrinsics_path=inputs.camera.path,
        object_specs=tuple(
            (track.name, track.mesh_path, track.poses_dir) for track in inputs.objects
        ),
        lineage_manifest=destination,
    )
    assert accepted.upstream_lineage.manifest == manifest
    assert accepted.upstream_lineage.verified_source_claim["uses_ground_truth"] is False


def test_execution_identity_requires_an_immutable_docker_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_inputs(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="immutable sha256"):
        render_v2d_mesh_pose_occluder(
            inputs,
            tmp_path / "output",
            renderer_image="v2d_foundation_pose:test",
            renderer_image_id="v2d_foundation_pose:test",
            renderer_factory=lambda _inputs, _platform: _FakeDepthRenderer(),
        )
