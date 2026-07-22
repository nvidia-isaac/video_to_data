from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from inpainting.adapters import sharpa_from_tracking, video2data_tracking
from inpainting.contracts import (
    TRACKING_SCHEMA,
    ContractError,
    VideoGeometry,
    validate_robot_trajectory_file,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _stable_sharpa_snapshots(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict]:
    implementation = root / "retarget.py"
    implementation.write_text("# pinned retarget implementation\n")
    asset = root / "hand.stl"
    asset.write_bytes(b"solid hand")
    implementation_artifacts = {"retarget.py": _artifact(implementation)}
    asset_artifacts = {"meshes/hand.stl": _artifact(asset)}
    monkeypatch.setattr(
        sharpa_from_tracking,
        "_implementation_artifacts",
        lambda: implementation_artifacts,
    )
    monkeypatch.setattr(
        sharpa_from_tracking,
        "sharpa_asset_artifacts",
        lambda _: asset_artifacts,
    )
    return implementation_artifacts, asset_artifacts


def _write_wilor_generation(raw_dir: Path, source_video: Path, image_id: str) -> Path:
    frame_paths = sorted(
        path for path in raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")
    )
    outputs = video2data_tracking._aggregate_files(frame_paths)
    outputs["files"] = {path.name: _artifact(path) for path in frame_paths}
    skeleton = {
        "execution_environment": {"container_image_id": image_id},
        "source_revisions": {
            "wilor_mini_commit": video2data_tracking.WILOR_SOURCE_COMMIT,
            "huggingface_revision": video2data_tracking.WILOR_HF_REVISION,
        },
        "sources": {"video": _artifact(source_video), "bboxes": None},
        "weights": {
            "MANO_RIGHT.pkl": {"size_bytes": 1, "sha256": "f" * 64},
            **{
                name: {"size_bytes": 1, "sha256": digest}
                for name, digest in video2data_tracking.PUBLIC_WEIGHT_SHA256.items()
            },
        },
        "implementation_sources": {
            "video_to_hands.py": {"size_bytes": 1, "sha256": "e" * 64}
        },
        "parameters": {"inference_mode": "detector"},
    }
    static_identity = video2data_tracking._wilor_static_identity(skeleton)
    names = [path.name for path in frame_paths]
    manifest = {
        "schema_version": video2data_tracking.WILOR_RUN_GENERATION_SCHEMA,
        "state": "complete",
        **skeleton,
        "static_identity": static_identity,
        "expected_frames": {"count": len(names), "filenames": names},
        "generation_id": video2data_tracking._wilor_generation_id(
            static_identity, names
        ),
        "outputs": outputs,
    }
    path = raw_dir / video2data_tracking.WILOR_RUN_GENERATION_FILENAME
    path.write_text(json.dumps(manifest))
    return path


def _write_enriched_tracking(root: Path, coordinate_frame: str = "camera") -> Path:
    path = root / "tracking.npz"
    frame_count = 2
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("phantom"),
        "coordinate_frame": np.asarray(coordinate_frame),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
    }
    for side in ("left", "right"):
        joints = np.zeros((frame_count, 21, 3), dtype=np.float32)
        joints[:, 0, 0] = [1.0, 2.0]
        quaternions = np.zeros((frame_count, 21, 4), dtype=np.float32)
        quaternions[..., 0] = 1.0
        arrays[f"{side}_valid"] = np.ones(frame_count, dtype=np.bool_)
        arrays[f"{side}_wrist_position"] = joints[:, 0].copy()
        arrays[f"{side}_wrist_wxyz"] = quaternions[:, 0].copy()
        arrays[f"{side}_joints_3d"] = joints
        arrays[f"{side}_joints_wxyz"] = quaternions
    np.savez_compressed(path, **arrays)
    return path


def _write_phantom_metadata(path: Path, tracking: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "v2d.inpainting.phantom-run/v1",
                "state": "complete",
                "sequence_id": "phantom-test",
                "tracker": "phantom",
                "coordinate_frame": "camera",
                "geometry": {
                    "frame_count": 2,
                    "width": 100,
                    "height": 50,
                    "fps": 29.97,
                },
                "outputs": {
                    "tracking": {
                        "filename": tracking.name,
                        "bytes": tracking.stat().st_size,
                        "sha256": _sha256(tracking),
                    }
                },
            }
        )
    )


def test_tracking_metadata_strictly_allows_v2d_and_phantom_layouts(
    tmp_path: Path,
) -> None:
    tracking = _write_enriched_tracking(tmp_path)
    phantom = tmp_path / "phantom.json"
    _write_phantom_metadata(phantom, tracking)
    loaded = sharpa_from_tracking._load_tracking_metadata(phantom, tracking)
    assert loaded["tracker"] == "phantom"

    v2d = tmp_path / "v2d.json"
    v2d.write_text(
        json.dumps(
            {
                "schema_version": "v2d.inpainting.video2data-tracking-stage/v1",
                "state": "complete",
                "tracker": "phantom",
                "coordinate_frame": "camera",
                "video": {
                    "frame_count": 2,
                    "width": 100,
                    "height": 50,
                    "fps": 29.97,
                },
                "tracking": {
                    "size_bytes": tracking.stat().st_size,
                    "sha256": _sha256(tracking),
                },
            }
        )
    )
    assert sharpa_from_tracking._load_tracking_metadata(v2d, tracking)[
        "schema_version"
    ].endswith("/v1")

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": "v2d.inpainting.unverified/v1",
                "state": "complete",
            }
        )
    )
    with pytest.raises(ContractError, match="Unsupported tracking metadata schema"):
        sharpa_from_tracking._load_tracking_metadata(bad, tracking)


def test_sharpa_stage_inherits_phantom_tracker_geometry_and_camera_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeBackend:
        identity = "fake-sharpa"

        def __init__(self, **_):
            self.diagnostics = {
                side: {
                    "accepted_frames": 2,
                    "rejected_frames": 0,
                    "median_frame_task_error_m": 0.001,
                    "p95_frame_task_error_m": 0.001,
                    "max_frame_task_error_m": 0.001,
                }
                for side in ("left", "right")
            }

        def retarget(self, *, joints, joints_wxyz, valid, mano_to_robot_scale):
            assert mano_to_robot_scale == pytest.approx(1.2)
            return {
                side: {
                    "valid": valid[side].copy(),
                    "wrist_position": joints[side][:, 0].copy(),
                    "wrist_wxyz": joints_wxyz[side][:, 0].copy(),
                    "finger_joints": np.zeros((2, 2), dtype=np.float32),
                    "finger_joint_names": np.asarray(["a", "b"]),
                }
                for side in ("left", "right")
            }

    tracking = _write_enriched_tracking(tmp_path)
    metadata_path = tmp_path / "run_metadata.json"
    _write_phantom_metadata(metadata_path, tracking)
    monkeypatch.setattr(sharpa_from_tracking, "ExistingSharpaBackend", FakeBackend)
    monkeypatch.setattr(sharpa_from_tracking, "sharpa_asset_blockers", lambda _: [])
    implementation, assets = _stable_sharpa_snapshots(tmp_path, monkeypatch)
    image_id = "sha256:" + "a" * 64
    result = sharpa_from_tracking.execute_sharpa(
        tracking=tracking,
        tracking_metadata=metadata_path,
        output_dir=tmp_path / "output",
        robot_assets_dir=tmp_path / "assets",
        sharpa_image_id=image_id,
        device="cpu",
        mano_to_robot_scale=1.2,
        max_frame_task_error_m=0.07,
        overwrite=False,
    )
    assert result["tracker"] == "phantom"
    assert result["coordinate_frame"] == "camera"
    assert result["video"]["fps"] == pytest.approx(29.97)
    assert result["execution_environment"]["container_image_id"] == image_id
    assert result["implementation_sources"] == implementation
    assert result["robot_assets"]["files"] == assets
    trajectory_path = tmp_path / "output" / "robot_trajectory.npz"
    validate_robot_trajectory_file(trajectory_path, expected_frames=2)
    with np.load(trajectory_path, allow_pickle=False) as data:
        assert str(data["coordinate_frame"].item()) == "camera"


def test_sharpa_stage_refuses_complete_commit_when_any_observation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RejectingBackend:
        identity = "rejecting-sharpa"

        def __init__(self, **_):
            self.diagnostics = {}

        def retarget(self, *, joints, joints_wxyz, valid, mano_to_robot_scale):
            results = {}
            for side in ("left", "right"):
                accepted = valid[side].copy()
                if side == "left":
                    accepted[1] = False
                results[side] = {
                    "valid": accepted,
                    "wrist_position": joints[side][:, 0].copy(),
                    "wrist_wxyz": joints_wxyz[side][:, 0].copy(),
                    "finger_joints": np.zeros((2, 2), dtype=np.float32),
                    "finger_joint_names": np.asarray(["a", "b"]),
                }
            return results

    tracking = _write_enriched_tracking(tmp_path)
    metadata_path = tmp_path / "run_metadata.json"
    _write_phantom_metadata(metadata_path, tracking)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(sharpa_from_tracking, "ExistingSharpaBackend", RejectingBackend)
    monkeypatch.setattr(sharpa_from_tracking, "sharpa_asset_blockers", lambda _: [])
    _stable_sharpa_snapshots(tmp_path, monkeypatch)

    with pytest.raises(
        sharpa_from_tracking.AdapterError,
        match=r"left: accepted 1/2 input-valid frames.*rejected indices=\[1\]",
    ):
        sharpa_from_tracking.execute_sharpa(
            tracking=tracking,
            tracking_metadata=metadata_path,
            output_dir=output_dir,
            robot_assets_dir=tmp_path / "assets",
            sharpa_image_id="sha256:" + "a" * 64,
            device="cpu",
            mano_to_robot_scale=1.2,
            max_frame_task_error_m=0.07,
            overwrite=False,
        )
    assert not output_dir.exists()


def test_sharpa_rehashes_assets_before_atomic_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets_root = tmp_path / "assets"
    xml_dir = assets_root / "xmls" / "sharpawave"
    xml_dir.mkdir(parents=True)
    mesh_paths: list[Path] = []
    for side in ("left", "right"):
        mesh_dir = assets_root / "meshes" / "sharpa_wave" / side
        mesh_dir.mkdir(parents=True)
        mesh = mesh_dir / f"{side}.stl"
        mesh.write_bytes(b"solid original")
        mesh_paths.append(mesh)
        (xml_dir / f"{side}_sharpawave.xml").write_text(
            '<mujoco><compiler meshdir="../../meshes/sharpa_wave/'
            f'{side}"/><asset><mesh file="{side}.stl"/></asset></mujoco>'
        )

    implementation = tmp_path / "implementation.py"
    implementation.write_text("# stable\n")
    monkeypatch.setattr(
        sharpa_from_tracking,
        "_implementation_artifacts",
        lambda: {"implementation.py": _artifact(implementation)},
    )

    class MutatingBackend:
        identity = "mutating-backend"

        def __init__(self, **_):
            self.diagnostics = {}

        def retarget(self, *, joints, joints_wxyz, valid, mano_to_robot_scale):
            mesh_paths[0].write_bytes(b"solid changed")
            return {
                side: {
                    "valid": valid[side].copy(),
                    "wrist_position": joints[side][:, 0].copy(),
                    "wrist_wxyz": joints_wxyz[side][:, 0].copy(),
                    "finger_joints": np.zeros((2, 2), dtype=np.float32),
                    "finger_joint_names": np.asarray(["a", "b"]),
                }
                for side in ("left", "right")
            }

    monkeypatch.setattr(sharpa_from_tracking, "ExistingSharpaBackend", MutatingBackend)
    tracking = _write_enriched_tracking(tmp_path)
    metadata_path = tmp_path / "run_metadata.json"
    _write_phantom_metadata(metadata_path, tracking)
    output_dir = tmp_path / "output"
    with pytest.raises(
        sharpa_from_tracking.AdapterError,
        match="Sharpa XML/mesh assets changed",
    ):
        sharpa_from_tracking.execute_sharpa(
            tracking=tracking,
            tracking_metadata=metadata_path,
            output_dir=output_dir,
            robot_assets_dir=assets_root,
            sharpa_image_id="sha256:" + "a" * 64,
            device="cpu",
            mano_to_robot_scale=1.2,
            max_frame_task_error_m=0.07,
            overwrite=False,
        )
    assert not (output_dir / "robot_trajectory.npz").exists()
    assert not (output_dir / "robot_trajectory.json").exists()


def test_raw_wilor_generation_refuses_extra_or_tampered_frame(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = raw / "000000.json"
    frame.write_text("[]")
    image_id = "sha256:" + "b" * 64
    _write_wilor_generation(raw, source, image_id)
    validated = video2data_tracking._validate_wilor_run_generation(
        json_dir=raw,
        source_video=source,
        expected_frames=1,
        expected_image_id=image_id,
    )
    assert validated["generation_id"].startswith("sha256:")

    extra = raw / "999999.json"
    extra.write_text("[]")
    with pytest.raises(
        video2data_tracking.AdapterError, match="exactly its committed generation"
    ):
        video2data_tracking._validate_wilor_run_generation(
            json_dir=raw,
            source_video=source,
            expected_frames=1,
            expected_image_id=image_id,
        )
    extra.unlink()
    frame.write_text('[{"stale": true}]')
    with pytest.raises(video2data_tracking.AdapterError, match=r"output .* mismatch"):
        video2data_tracking._validate_wilor_run_generation(
            json_dir=raw,
            source_video=source,
            expected_frames=1,
            expected_image_id=image_id,
        )


def test_tracking_stage_records_complete_raw_wilor_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeMano:
        identity = "fake-mano"

        def __init__(self, *_args, **_kwargs):
            pass

        def forward(self, *, global_orient, **_kwargs):
            count = len(global_orient)
            quaternions = np.zeros((count, 21, 4), dtype=np.float32)
            quaternions[..., 0] = 1.0
            return {
                "joints": np.zeros((count, 21, 3), dtype=np.float32),
                "joints_wxyz": quaternions,
                "vertices": np.zeros((count, 778, 3), dtype=np.float32),
            }

    def record(is_right: bool) -> dict:
        return {
            "is_right": is_right,
            "score": 1.0,
            "bbox": {"x0": 10.0, "y0": 5.0, "x1": 30.0, "y1": 35.0},
            "mano": {
                "betas": [0.0] * 10,
                "global_orient": [0.0] * 3,
                "hand_pose": [0.0] * 45,
            },
            "camera": {
                "pred_cam_t_full": [0.0, 0.0, 1.0],
                "scaled_focal_length": 100.0,
            },
            "image_size": [100, 50],
        }

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video-bytes")
    raw_dir = tmp_path / "wilor_raw"
    raw_dir.mkdir()
    (raw_dir / "000000.json").write_text(json.dumps([record(False), record(True)]))
    intrinsic = tmp_path / "intrinsic.txt"
    np.savetxt(
        intrinsic,
        np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]]),
    )
    extrinsic = tmp_path / "extrinsic.npy"
    np.save(extrinsic, np.eye(4)[None])
    mano_root = tmp_path / "mano"
    (mano_root / "models").mkdir(parents=True)
    for side in ("LEFT", "RIGHT"):
        (mano_root / "models" / f"MANO_{side}.pkl").write_bytes(
            f"licensed-{side}".encode()
        )
    monkeypatch.setattr(
        video2data_tracking,
        "probe_video",
        lambda _: VideoGeometry(frame_count=1, width=100, height=50, fps=29.97),
    )
    monkeypatch.setattr(video2data_tracking, "ManoTorchBackend", FakeMano)
    monkeypatch.setattr(
        video2data_tracking,
        "_public_weight_artifacts",
        lambda _: ({"detector.pt": {"sha256": "a" * 64}}, []),
    )
    image_id = "sha256:" + "b" * 64
    _write_wilor_generation(raw_dir, source_video, image_id)
    output = tmp_path / "output"
    metadata = video2data_tracking.execute_tracking(
        result_dir=None,
        wilor_json_dir=raw_dir,
        taco_intrinsic=intrinsic,
        taco_extrinsic=extrinsic,
        source_video=source_video,
        output_dir=output,
        mano_model_dir=mano_root,
        public_weights_dir=tmp_path / "weights",
        wilor_image_id=image_id,
        device="cpu",
        sequence_id="synthetic",
        allow_static_camera=False,
        overwrite=False,
    )
    assert metadata["schema_version"] == video2data_tracking.STAGE_SCHEMA
    assert metadata["schema_version"].endswith("/v2")
    provenance = metadata["provenance"]
    assert provenance["raw_wilor_json"]["file_count"] == 1
    assert len(provenance["raw_wilor_json"]["aggregate_sha256"]) == 64
    assert provenance["raw_wilor_generation"]["container_image_id"] == image_id
    assert provenance["raw_wilor_generation"]["generation_id"].startswith("sha256:")
    assert provenance["raw_wilor_generation"]["manifest"]["sha256"] == _sha256(
        raw_dir / video2data_tracking.WILOR_RUN_GENERATION_FILENAME
    )
    assert provenance["source_video"]["sha256"] == _sha256(source_video)
    assert provenance["taco_camera"]["intrinsic"]["sha256"] == _sha256(intrinsic)
    assert provenance["taco_camera"]["world_to_camera"]["sha256"] == _sha256(extrinsic)
    assert provenance["mano_models"]["left"]["sha256"] == _sha256(
        mano_root / "models" / "MANO_LEFT.pkl"
    )
    assert provenance["wilor"]["container_image_id"] == image_id
    assert (
        provenance["wilor"]["source_commit"] == video2data_tracking.WILOR_SOURCE_COMMIT
    )
    assert (
        provenance["wilor"]["huggingface_revision"]
        == video2data_tracking.WILOR_HF_REVISION
    )
    assert metadata["tracking"]["sha256"] == _sha256(output / "tracking.npz")
