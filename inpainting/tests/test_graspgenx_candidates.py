from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inpainting import graspgenx_candidates as candidates


trimesh = pytest.importorskip("trimesh")


SWEEP_ARGUMENTS: dict[str, Any] = {
    "extents_open": [0.10, 0.02, 0.04],
    "offset_open": [0.0, 0.0, 0.06],
    "extents_mid": [0.05, 0.02, 0.04],
    "offset_mid": [0.0, 0.0, 0.06],
    "fingertip_depth": 0.08,
    "gripper_type": "parallel_2f",
    "gripper_name": "test_parallel_jaw",
}


class FakeProvider:
    def __init__(self, transforms: np.ndarray | None = None, scores=None) -> None:
        if transforms is None:
            transforms = np.repeat(np.eye(4)[None], 3, axis=0)
            transforms[:, :3, 3] = [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [0.00, 0.02, 0.00],
            ]
        self.transforms = np.asarray(transforms)
        self.scores = np.asarray(
            [0.2, 0.9, 0.5] if scores is None else scores
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        copied = dict(kwargs)
        copied["point_cloud_object_centered"] = np.asarray(
            kwargs["point_cloud_object_centered"]
        ).copy()
        self.calls.append(copied)
        return self.transforms.copy(), self.scores.copy()


@pytest.fixture
def fake_runtime(tmp_path: Path) -> tuple[Path, Path]:
    graspgenx_root = tmp_path / "GraspGenX"
    package = graspgenx_root / "graspgenx"
    package.mkdir(parents=True)
    (package / "grasp_server.py").write_text("# fake\n", encoding="utf-8")
    (package / "__init__.py").write_text(
        '__version__ = "test-version"\n', encoding="utf-8"
    )
    (graspgenx_root / "pyproject.toml").write_text(
        '[project]\nname = "graspgenx"\nversion = "9.8.7"\n',
        encoding="utf-8",
    )

    checkpoint_root = tmp_path / "checkpoints" / "release"
    for role, epoch in (("gen", 736), ("dis", 1056)):
        directory = checkpoint_root / role
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(
            f"role: {role}\n", encoding="utf-8"
        )
        (directory / f"epoch_{epoch}.pth").write_bytes(
            f"{role}-checkpoint".encode()
        )
    return graspgenx_root, checkpoint_root


def _write_mesh(path: Path) -> None:
    box = trimesh.creation.box(extents=[0.12, 0.04, 0.03])
    transform = np.eye(4)
    transform[:3, 3] = [0.25, -0.13, 0.07]
    if path.suffix == ".glb":
        scene = trimesh.Scene()
        scene.add_geometry(box, node_name="metric_box", transform=transform)
        path.write_bytes(scene.export(file_type="glb"))
    else:
        box.apply_transform(transform)
        box.export(path)


def _generate(
    mesh: Path,
    output: Path,
    runtime: tuple[Path, Path],
    provider: FakeProvider,
    **kwargs,
) -> candidates.CandidateArtifact:
    graspgenx_root, checkpoint_root = runtime
    return candidates.generate_graspgenx_candidates(
        mesh,
        output,
        **SWEEP_ARGUMENTS,
        graspgenx_root=graspgenx_root,
        checkpoint_root=checkpoint_root,
        seed=123,
        num_grasps=3,
        top_k=2,
        num_sample_points=512,
        provider=provider,
        **kwargs,
    )


@pytest.mark.parametrize("suffix", [".obj", ".ply", ".glb"])
def test_metric_mesh_formats_are_loaded_and_sampled_deterministically(
    tmp_path: Path,
    fake_runtime: tuple[Path, Path],
    suffix: str,
) -> None:
    mesh = tmp_path / f"object{suffix}"
    _write_mesh(mesh)
    first_provider = FakeProvider()
    second_provider = FakeProvider()

    first = _generate(
        mesh, tmp_path / f"first_{suffix[1:]}.npz", fake_runtime, first_provider
    )
    second = _generate(
        mesh, tmp_path / f"second_{suffix[1:]}.npz", fake_runtime, second_provider
    )

    first_points = first_provider.calls[0]["point_cloud_object_centered"]
    second_points = second_provider.calls[0]["point_cloud_object_centered"]
    assert first_points.shape == (512, 3)
    assert first_points.dtype == np.float32
    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_allclose(first_points.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_array_equal(
        first.object_to_gripper_base, second.object_to_gripper_base
    )
    np.testing.assert_array_equal(first.confidence, second.confidence)
    assert first.provenance["sampling"]["centered_points_sha256"] == (
        second.provenance["sampling"]["centered_points_sha256"]
    )


def test_glb_generation_writes_exact_contract_and_full_provenance(
    tmp_path: Path,
    fake_runtime: tuple[Path, Path],
) -> None:
    mesh = tmp_path / "object.glb"
    output = tmp_path / "candidates.npz"
    _write_mesh(mesh)
    provider = FakeProvider()

    artifact = _generate(mesh, output, fake_runtime, provider)

    assert artifact.npz_path == output
    assert artifact.provenance_path == output.with_suffix(".json")
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["sweep_volume"]["gripper_type"] == 0
    assert call["sweep_volume"]["fingertip_depth"] == pytest.approx(0.08)
    assert call["seed"] == 123
    assert call["num_grasps"] == 3

    with np.load(output, allow_pickle=False) as archive:
        assert archive.files == ["object_to_gripper_base", "confidence"]
        transforms = archive["object_to_gripper_base"]
        confidence = archive["confidence"]
    assert transforms.shape == (2, 4, 4)
    assert transforms.dtype == np.float32
    assert confidence.shape == (2,)
    assert confidence.dtype == np.float32
    np.testing.assert_array_equal(confidence, np.array([0.9, 0.5], np.float32))

    sample_mean = np.asarray(
        artifact.provenance["sampling"]["sample_mean_object_m"]
    )
    np.testing.assert_allclose(
        transforms[:, :3, 3] - sample_mean,
        [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]],
        atol=2e-7,
    )

    provenance = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == candidates.SCHEMA_VERSION
    assert provenance["adapter"]["version"] == candidates.ADAPTER_VERSION
    assert provenance["graspgenx"]["declared_version"] == "9.8.7"
    assert provenance["input"]["mesh"]["sha256"] == hashlib.sha256(
        mesh.read_bytes()
    ).hexdigest()
    assert provenance["input"]["mesh_details"]["format"] == "glb"
    assert provenance["sampling"]["seed"] == 123
    assert provenance["gripper"]["gripper_name"] == "test_parallel_jaw"
    assert provenance["inference"]["requested_top_k"] == 2
    assert provenance["output"]["npz"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert (
        provenance["checkpoints"]["gen"]["config"]["sha256"]
        == hashlib.sha256(
            (fake_runtime[1] / "gen" / "config.yaml").read_bytes()
        ).hexdigest()
    )
    assert (
        provenance["checkpoints"]["dis"]["checkpoint"]["sha256"]
        == hashlib.sha256(
            (fake_runtime[1] / "dis" / "epoch_1056.pth").read_bytes()
        ).hexdigest()
    )


def test_stable_confidence_sort_preserves_equal_score_provider_order() -> None:
    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[:, 0, 3] = [1.0, 2.0, 3.0]
    validated, scores = candidates.validate_candidates(
        transforms, [0.5, 0.5, 0.2]
    )
    order = np.argsort(-scores, kind="stable")
    np.testing.assert_array_equal(validated[order, 0, 3], [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("mutate_transform", "scores", "message"),
    [
        (
            lambda values: values.__setitem__((0, 3, 0), 0.1),
            [0.5],
            "homogeneous row",
        ),
        (
            lambda values: values.__setitem__((0, 0, 0), 2.0),
            [0.5],
            "orthonormal",
        ),
        (
            lambda values: values.__setitem__((0, 0, 0), -1.0),
            [0.5],
            "determinant",
        ),
        (
            lambda values: None,
            [float("nan")],
            "non-finite",
        ),
        (
            lambda values: None,
            [1.1],
            r"\[0, 1\]",
        ),
    ],
)
def test_candidate_validation_rejects_non_rigid_poses_and_invalid_scores(
    mutate_transform,
    scores,
    message: str,
) -> None:
    transforms = np.eye(4)[None]
    mutate_transform(transforms)
    with pytest.raises(candidates.CandidateValidationError, match=message):
        candidates.validate_candidates(transforms, scores)


def test_candidate_validation_rejects_shape_and_count_mismatches() -> None:
    with pytest.raises(candidates.CandidateValidationError, match=r"\(N, 4, 4\)"):
        candidates.validate_candidates(np.eye(4), [0.2])
    with pytest.raises(candidates.CandidateValidationError, match="counts differ"):
        candidates.validate_candidates(
            np.repeat(np.eye(4)[None], 2, axis=0), [0.2]
        )
    with pytest.raises(candidates.CandidateValidationError, match=r"shape \(N,\)"):
        candidates.validate_candidates(np.eye(4)[None], [[0.2]])
    with pytest.raises(candidates.CandidateValidationError, match="no grasp"):
        candidates.validate_candidates(
            np.empty((0, 4, 4)), np.empty((0,))
        )


def test_sweep_volume_validation_is_strict() -> None:
    with pytest.raises(candidates.CandidateValidationError, match="strictly positive"):
        candidates.SweepVolume.create(
            **{**SWEEP_ARGUMENTS, "extents_open": [0.1, 0.0, 0.2]}
        )
    with pytest.raises(candidates.CandidateValidationError, match="unknown"):
        candidates.SweepVolume.create(
            **{**SWEEP_ARGUMENTS, "gripper_type": "magic"}
        )
    with pytest.raises(candidates.CandidateValidationError, match="fingertip"):
        candidates.SweepVolume.create(
            **{**SWEEP_ARGUMENTS, "fingertip_depth": float("nan")}
        )


def test_non_overwrite_fails_before_loading_or_provider_call(
    tmp_path: Path,
    fake_runtime: tuple[Path, Path],
) -> None:
    mesh = tmp_path / "object.glb"
    _write_mesh(mesh)
    output = tmp_path / "candidates.npz"
    original = b"owned by another run"
    output.write_bytes(original)
    provider = FakeProvider()

    with pytest.raises(candidates.ArtifactExistsError):
        _generate(mesh, output, fake_runtime, provider)

    assert output.read_bytes() == original
    assert provider.calls == []
    assert not output.with_suffix(".json").exists()


def test_existing_provenance_blocks_pair_without_publishing_npz(
    tmp_path: Path,
    fake_runtime: tuple[Path, Path],
) -> None:
    mesh = tmp_path / "object.glb"
    _write_mesh(mesh)
    output = tmp_path / "candidates.npz"
    provenance = output.with_suffix(".json")
    provenance.write_text('{"owner": "other"}\n', encoding="utf-8")
    provider = FakeProvider()

    with pytest.raises(candidates.ArtifactExistsError):
        _generate(mesh, output, fake_runtime, provider)

    assert not output.exists()
    assert provenance.read_text(encoding="utf-8") == '{"owner": "other"}\n'
    assert provider.calls == []


def test_invalid_provider_output_leaves_no_partial_artifacts(
    tmp_path: Path,
    fake_runtime: tuple[Path, Path],
) -> None:
    mesh = tmp_path / "object.glb"
    _write_mesh(mesh)
    output = tmp_path / "candidates.npz"
    invalid = np.eye(4)[None]
    invalid[0, 0, 0] = 4.0

    with pytest.raises(candidates.CandidateValidationError, match="orthonormal"):
        _generate(
            mesh,
            output,
            fake_runtime,
            FakeProvider(invalid, [0.5]),
        )

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_pair_commit_rolls_back_npz_if_json_publish_races(
    tmp_path: Path,
    fake_runtime: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = tmp_path / "object.glb"
    _write_mesh(mesh)
    output = tmp_path / "candidates.npz"
    real_link = candidates.os.link
    link_calls = 0

    def racing_link(source, target):
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise FileExistsError(target)
        return real_link(source, target)

    monkeypatch.setattr(candidates.os, "link", racing_link)
    with pytest.raises(candidates.ArtifactExistsError, match="appeared"):
        _generate(mesh, output, fake_runtime, FakeProvider())

    assert link_calls == 2
    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_overwrite_is_explicit_and_replaces_both_artifacts(
    tmp_path: Path,
    fake_runtime: tuple[Path, Path],
) -> None:
    mesh = tmp_path / "object.glb"
    _write_mesh(mesh)
    output = tmp_path / "candidates.npz"
    _generate(mesh, output, fake_runtime, FakeProvider())
    first_provenance = output.with_suffix(".json").read_bytes()

    replacement = FakeProvider(scores=[0.1, 0.3, 0.8])
    artifact = _generate(
        mesh,
        output,
        fake_runtime,
        replacement,
        overwrite=True,
    )

    np.testing.assert_array_equal(
        artifact.confidence, np.array([0.8, 0.3], dtype=np.float32)
    )
    assert output.with_suffix(".json").read_bytes() != first_provenance
