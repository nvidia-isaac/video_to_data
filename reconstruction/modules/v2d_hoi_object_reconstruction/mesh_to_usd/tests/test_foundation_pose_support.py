import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

import foundation_pose_support as foundation_pose
import v2d.foundation_pose.docker.run_mv_videos_to_poses as foundation_pose_launcher
from foundation_pose_support import (
    DEFAULT_FOUNDATION_POSE_CONFIG,
    FOUNDATION_POSE_TRACKING_METADATA_NAME,
    run_foundation_pose_support,
)


def _write_inputs(tmp_path):
    sequence = tmp_path / "sequence"
    for relative in (
        "edex",
        "images",
        "depth",
        "object_masks",
    ):
        (sequence / relative).mkdir(parents=True, exist_ok=True)
    (sequence / "ground_plane.json").write_text(
        json.dumps({"plane": [0.0, 0.0, 1.0, 0.0]}),
        encoding="utf-8",
    )
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "output_aligned.glb"
    target.write_bytes(b"target mesh")
    (target_dir / "output_symmetry.json").write_text(
        json.dumps({"symmetries_discrete": []}),
        encoding="utf-8",
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    return sequence, target, weights


def _write_runner_output(
    output,
    pose_count,
    *,
    frame_start=0,
    frame_end_exclusive=None,
):
    output = Path(output)
    camera_names = [
        "front_stereo_camera_left",
        "back_stereo_camera_left",
    ]
    frame_end = (
        frame_start + pose_count
        if frame_end_exclusive is None
        else frame_end_exclusive
    )
    poses = np.repeat(np.eye(4)[None, :, :], pose_count, axis=0)
    np.save(output / "poses.npy", poses)
    (output / FOUNDATION_POSE_TRACKING_METADATA_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "source_frame_start": frame_start,
                "source_frame_end_exclusive": frame_end,
                "pose_count": pose_count,
                "camera_names": camera_names,
                "registration_camera_names": camera_names,
                "highest_visibility_registration_camera": camera_names[0],
                "registration_visible_ratios": {
                    camera_names[0]: 0.8,
                    camera_names[1]: 0.6,
                },
                "tracking_camera_frame_counts": {
                    camera_names[0]: pose_count,
                    camera_names[1]: pose_count,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_registration_failure(
    output,
    *,
    frame_start,
    error="registration failed",
):
    output = Path(output)
    (output / FOUNDATION_POSE_TRACKING_METADATA_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "source_frame_start": frame_start,
                "source_frame_end_exclusive": frame_start,
                "pose_count": 0,
                "camera_names": ["front_stereo_camera_left"],
                "error": error,
            }
        ),
        encoding="utf-8",
    )


def test_foundation_pose_support_runs_only_through_selected_window(tmp_path):
    sequence, target, weights = _write_inputs(tmp_path)
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        _write_runner_output(
            kwargs["output_dir"],
            20,
            frame_start=kwargs["frame_start"],
        )

    result = run_foundation_pose_support(
        str(sequence),
        str(target),
        str(weights),
        str(tmp_path / "output"),
        frame_end_exclusive=20,
        runner=runner,
    )

    assert len(calls) == 1
    assert calls[0]["frame_start"] == 0
    assert calls[0]["frame_end_exclusive"] == 20
    assert calls[0]["clamp_frame_end_exclusive"] is False
    assert "registration_max_attempts" not in calls[0]
    assert "registration_attempt_stride" not in calls[0]
    assert "minimum_output_frames" not in calls[0]
    assert calls[0]["camera_params_path"] == str(
        (sequence / "edex").resolve()
    )
    assert calls[0]["rgb_dir"] == str((sequence / "images").resolve())
    assert calls[0]["depth_dir"] == str((sequence / "depth").resolve())
    assert calls[0]["mask_dir"] == str(
        (sequence / "object_masks").resolve()
    )
    assert calls[0]["config_path"] == str(
        DEFAULT_FOUNDATION_POSE_CONFIG.resolve()
    )
    assert "exported_sequence_dir" not in calls[0]
    assert result["status"] == "completed"
    assert result["pose_count"] == 20
    assert len(result["poses_file_sha256"]) == 64
    assert result["camera_provenance"] == {
        "mode": "multi-view-foundation-pose",
        "camera_names": [
            "front_stereo_camera_left",
            "back_stereo_camera_left",
        ],
        "registration_frame": 0,
        "registration_camera_names": [
            "front_stereo_camera_left",
            "back_stereo_camera_left",
        ],
        "highest_visibility_registration_camera": (
            "front_stereo_camera_left"
        ),
        "registration_visible_ratios": {
            "front_stereo_camera_left": 0.8,
            "back_stereo_camera_left": 0.6,
        },
        "tracking_camera_frame_counts": {
            "front_stereo_camera_left": 20,
            "back_stereo_camera_left": 20,
        },
    }
    assert Path(result["output_poses"]).is_file()
    assert Path(result["report_file"]).is_file()


def test_foundation_pose_support_accepts_short_automatic_prefix(tmp_path):
    sequence, target, weights = _write_inputs(tmp_path)
    output = tmp_path / "output"
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        _write_runner_output(
            output,
            35,
            frame_start=kwargs["frame_start"],
        )

    result = run_foundation_pose_support(
        str(sequence),
        str(target),
        str(weights),
        str(output),
        frame_end_exclusive=90,
        allow_shorter_prefix=True,
        runner=runner,
    )

    assert calls[0]["frame_end_exclusive"] == 90
    assert calls[0]["clamp_frame_end_exclusive"] is True
    assert result["requested_frame_end_exclusive"] == 90
    assert result["frame_end_exclusive"] == 35
    assert result["pose_count"] == 35


def test_foundation_pose_support_uses_repository_default_weights(
    tmp_path,
    monkeypatch,
):
    sequence, target, weights = _write_inputs(tmp_path)
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(
        foundation_pose,
        "DEFAULT_FOUNDATION_POSE_WEIGHTS_DIR",
        weights,
    )

    def runner(**kwargs):
        calls.append(kwargs)
        _write_runner_output(
            output,
            5,
            frame_start=kwargs["frame_start"],
        )

    result = run_foundation_pose_support(
        str(sequence),
        str(target),
        None,
        str(output),
        frame_end_exclusive=5,
        runner=runner,
    )

    assert calls[0]["weights_dir"] == str(weights.resolve())
    assert result["weights_dir"] == str(weights.resolve())


def test_foundation_pose_support_accepts_path_template_config(tmp_path):
    sequence, target, weights = _write_inputs(tmp_path)
    output = tmp_path / "output"
    config = tmp_path / "custom_layout.yaml"
    config.write_text(
        'rgb_path_template: "${rgb_dir}/{cam_name}"\n'
        'depth_path_template: "${depth_dir}/{cam_name}"\n'
        'mask_path_template: "${mask_dir}/{cam_name}"\n',
        encoding="utf-8",
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        _write_runner_output(
            output,
            5,
            frame_start=kwargs["frame_start"],
        )

    result = run_foundation_pose_support(
        str(sequence),
        str(target),
        str(weights),
        str(output),
        frame_end_exclusive=5,
        config_path=str(config),
        runner=runner,
    )

    assert calls[0]["config_path"] == str(config.resolve())
    assert result["config_path"] == str(config.resolve())


def test_cross_mesh_config_disables_recovery_and_repair_only():
    base = OmegaConf.load(foundation_pose_launcher._LIB_CONFIG)
    overlay = OmegaConf.load(DEFAULT_FOUNDATION_POSE_CONFIG)
    merged = OmegaConf.merge(base, overlay)

    assert base.recovery.enabled is True
    assert base.repair.enabled is True
    assert merged.recovery.enabled is False
    assert merged.repair.enabled is False
    assert merged.recovery.min_views == base.recovery.min_views
    assert merged.repair.arbitration == base.repair.arbitration
    merged.rgb_dir = "/rgb"
    assert merged.rgb_path_template == "/rgb/{cam_name}.h5"


def test_foundation_pose_launcher_keeps_general_path_interface(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_run_in_container(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        foundation_pose_launcher,
        "run_in_container",
        fake_run_in_container,
    )
    foundation_pose_launcher.run_mv_videos_to_poses(
        camera_params_path=str(tmp_path / "sequence" / "edex"),
        rgb_dir=str(tmp_path / "sequence" / "images"),
        depth_dir=str(tmp_path / "sequence" / "depth"),
        mask_dir=str(tmp_path / "sequence" / "object_masks"),
        mesh_path=str(tmp_path / "mesh" / "output_aligned.glb"),
        symmetry_path=str(tmp_path / "mesh" / "output_symmetry.json"),
        weights_dir=str(tmp_path / "weights" / "foundationpose"),
        output_dir=str(tmp_path / "output"),
        config_path=str(DEFAULT_FOUNDATION_POSE_CONFIG),
        frame_start=10,
        frame_end_exclusive=90,
        clamp_frame_end_exclusive=True,
    )

    assert set(captured["inputs"]) == {
        "weights_dir",
        "camera_params_path",
        "rgb_dir",
        "depth_dir",
        "mask_dir",
        "mesh_path",
        "symmetry_path",
        "config_path",
    }
    assert captured["extra_args"]["frame_start"] == 10
    assert captured["extra_args"]["frame_end_exclusive"] == 90
    assert captured["extra_args"]["clamp_frame_end_exclusive"] is True
    assert "registration_max_attempts" not in captured["extra_args"]
    assert "registration_attempt_stride" not in captured["extra_args"]
    assert "minimum_output_frames" not in captured["extra_args"]


def test_foundation_pose_support_preserves_late_registration_frame(tmp_path):
    sequence, target, weights = _write_inputs(tmp_path)
    output = tmp_path / "output"
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        if kwargs["frame_start"] < 10:
            _write_registration_failure(
                output,
                frame_start=kwargs["frame_start"],
            )
            raise RuntimeError("registration failed")
        _write_runner_output(
            output,
            80,
            frame_start=kwargs["frame_start"],
            frame_end_exclusive=90,
        )

    result = run_foundation_pose_support(
        str(sequence),
        str(target),
        str(weights),
        str(output),
        frame_end_exclusive=90,
        minimum_output_frames=30,
        runner=runner,
    )

    assert result["frame_start"] == 10
    assert result["frame_end_exclusive"] == 90
    assert result["pose_count"] == 80
    assert [call["frame_start"] for call in calls] == [0, 5, 10]
    assert [
        attempt["status"] for attempt in result["registration_attempts"]
    ] == ["failed", "failed", "completed"]
    assert all(
        attempt["retryable"]
        for attempt in result["registration_attempts"][:2]
    )
    assert len(result["tracking_metadata_file_sha256"]) == 64


def test_foundation_pose_support_does_not_retry_infrastructure_failure(tmp_path):
    sequence, target, weights = _write_inputs(tmp_path)
    output = tmp_path / "output"
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        raise OSError("Docker daemon is unavailable")

    with pytest.raises(OSError, match="Docker daemon"):
        run_foundation_pose_support(
            str(sequence),
            str(target),
            str(weights),
            str(output),
            frame_end_exclusive=90,
            minimum_output_frames=30,
            runner=runner,
        )

    assert len(calls) == 1
    report = json.loads(
        (output / "foundation_pose_support_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "OSError"
    assert report["registration_attempts"][0]["retryable"] is False


def test_foundation_pose_support_fails_on_short_pose_output(tmp_path):
    sequence, target, weights = _write_inputs(tmp_path)
    output = tmp_path / "output"

    def runner(**kwargs):
        _write_runner_output(
            output,
            4,
            frame_start=kwargs["frame_start"],
        )

    with pytest.raises(RuntimeError, match="all eligible"):
        run_foundation_pose_support(
            str(sequence),
            str(target),
            str(weights),
            str(output),
            frame_end_exclusive=5,
            registration_max_attempts=1,
            runner=runner,
        )

    report = json.loads(
        (output / "foundation_pose_support_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert "end exactly" in report["registration_attempts"][0]["error"]
    assert not (output / "poses.npy").exists()
    assert not (output / FOUNDATION_POSE_TRACKING_METADATA_NAME).exists()
