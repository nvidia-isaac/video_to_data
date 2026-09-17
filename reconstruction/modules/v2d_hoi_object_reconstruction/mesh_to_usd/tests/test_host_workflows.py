import json
import subprocess
from pathlib import Path

import pytest

import run_drop_test
import run_mesh_to_usd
import run_validate_usd


def _asset(tmp_path, name="object.usd"):
    asset = tmp_path / name
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    return asset


def _option(command, name):
    return command[command.index(name) + 1]


def test_drop_command_uses_safe_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("PRIVACY_CONSENT", raising=False)
    command = run_drop_test.build_drop_test_command(
        str(_asset(tmp_path)),
        str(tmp_path / "output"),
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
    )

    assert "--shm-size=2g" in command
    assert "--ipc=host" not in command
    assert _option(command, "--initial-pose") == "principal-6"
    assert _option(command, "--video-fps") == "30"
    assert _option(command, "--lift-height-ratio") == "0.25"
    assert _option(command, "--pose-settle-position-ratio") == "0.01"
    assert _option(command, "--pose-settle-angle") == "2.0"
    assert "--video" in command
    assert "--fail-if-not-standing" in command
    assert run_drop_test.IMAGE_NAME in command


def test_mesh_command_records_default_mass_provenance(tmp_path):
    command = run_mesh_to_usd.build_mesh_to_usd_command(
        str(_asset(tmp_path)),
        str(tmp_path / "output"),
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
    )

    assert _option(command, "--mass-kg") == "0.3"
    assert _option(command, "--mass-source") == "assumed_grounding_default"
    assert "--simplify-decomposition-source" in command


def test_mesh_command_can_restrict_isaac_sim_to_one_gpu(tmp_path):
    command = run_mesh_to_usd.build_mesh_to_usd_command(
        str(_asset(tmp_path)),
        str(tmp_path / "output"),
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
        gpu_device="0",
    )

    assert _option(command, "--gpus") == "device=0"


def test_mesh_command_rejects_multiple_gpu_devices(tmp_path):
    with pytest.raises(ValueError, match="exactly one GPU"):
        run_mesh_to_usd.build_mesh_to_usd_command(
            str(_asset(tmp_path)),
            str(tmp_path / "output"),
            accept_eula=True,
            gpu_device="0,1",
        )


def test_drop_command_rejects_invalid_support_tolerance(tmp_path):
    with pytest.raises(ValueError, match="thresholds"):
        run_drop_test.build_drop_test_command(
            str(_asset(tmp_path)),
            str(tmp_path / "output"),
            ground_tolerance=-0.01,
            accept_eula=True,
        )


def test_drop_command_accepts_support_refined_pca(tmp_path):
    command = run_drop_test.build_drop_test_command(
        str(_asset(tmp_path)),
        str(tmp_path / "output"),
        initial_pose="principal-6-support",
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
    )

    assert _option(command, "--initial-pose") == "principal-6-support"


def test_drop_command_accepts_strict_recorded_support(tmp_path):
    support = tmp_path / "support.json"
    support.write_text("{}", encoding="utf-8")
    command = run_drop_test.build_drop_test_command(
        str(_asset(tmp_path)),
        str(tmp_path / "output"),
        initial_pose="recorded-support",
        recorded_support_path=str(support),
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
    )

    assert _option(command, "--initial-pose") == "recorded-support"
    assert _option(command, "--recorded-support").endswith("/support.json")
    assert any("/data/recorded-support:ro" in value for value in command)


def test_drop_command_does_not_fall_back_without_recorded_support(tmp_path):
    with pytest.raises(ValueError, match="requires recorded_support_path"):
        run_drop_test.build_drop_test_command(
            str(_asset(tmp_path)),
            str(tmp_path / "output"),
            initial_pose="recorded-support",
            accept_eula=True,
        )


def test_drop_command_rejects_recorded_input_for_geometry_mode(tmp_path):
    support = tmp_path / "support.json"
    support.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="only be used"):
        run_drop_test.build_drop_test_command(
            str(_asset(tmp_path)),
            str(tmp_path / "output"),
            initial_pose="principal-6-support",
            recorded_support_path=str(support),
            accept_eula=True,
        )


def test_drop_command_rejects_video_rate_above_physics_rate(tmp_path):
    with pytest.raises(ValueError, match="video_fps"):
        run_drop_test.build_drop_test_command(
            str(_asset(tmp_path)),
            str(tmp_path / "output"),
            physics_hz=30,
            video_fps=60,
            accept_eula=True,
        )


def test_drop_runner_enforces_json_failure_even_with_zero_exit(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / run_drop_test.DROP_TEST_REPORT_NAME
    report_path.write_text('{"status": "stale"}', encoding="utf-8")

    def run_isaac_sim(*_args, **_kwargs):
        assert not report_path.exists()
        report_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "passed": False,
                    "standing": False,
                    "failure_reasons": ["optional_standing_requirement_not_met"],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(run_drop_test.subprocess, "run", run_isaac_sim)

    with pytest.raises(RuntimeError, match="optional_standing_requirement_not_met"):
        run_drop_test.run_drop_test(
            str(_asset(tmp_path)),
            str(output),
            accept_eula=True,
            cache_dir=str(tmp_path / "cache"),
        )


def test_drop_runner_rejects_stale_outputs(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / run_drop_test.DROP_TEST_REPORT_NAME
    video_path = output / "drop_test_stale.mp4"
    report_path.write_text('{"status": "passed", "passed": true}', encoding="utf-8")
    video_path.write_bytes(b"stale video")
    monkeypatch.setattr(
        run_drop_test.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )

    with pytest.raises(RuntimeError, match="exited with code 0"):
        run_drop_test.run_drop_test(
            str(_asset(tmp_path)),
            str(output),
            accept_eula=True,
            cache_dir=str(tmp_path / "cache"),
        )

    assert not report_path.exists()
    assert not video_path.exists()


def test_drop_runner_maps_combined_video_to_host(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / run_drop_test.DROP_TEST_REPORT_NAME
    video_path = output / "drop_test.mp4"

    def run_isaac_sim(*_args, **_kwargs):
        video_path.write_bytes(b"video")
        report_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "passed": True,
                    "standing": True,
                    "video_files": ["/data/output/drop_test.mp4"],
                    "pose_results": [
                        {
                            "pose_id": "principal_axis_0_positive",
                            "video_files": ["/data/output/drop_test.mp4"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(run_drop_test.subprocess, "run", run_isaac_sim)

    result = run_drop_test.run_drop_test(
        str(_asset(tmp_path)),
        str(output),
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
    )

    assert result["video_files"] == [str(video_path)]
    assert result["pose_results"][0]["video_files"] == [str(video_path)]
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["container_video_files"] == ["/data/output/drop_test.mp4"]


def test_mesh_runner_maps_container_output_to_host(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    rigid_object = output / "rigid_object.usd"
    visual_asset = output / "visual_asset.usd"
    report_path = output / run_mesh_to_usd.GENERATION_REPORT_NAME
    report_path.write_text('{"status": "stale"}', encoding="utf-8")

    def run_isaac_sim(*_args, **_kwargs):
        assert not report_path.exists()
        rigid_object.write_text("#usda 1.0\n", encoding="utf-8")
        visual_asset.write_text("#usda 1.0\n", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "status": "generated",
                    "output_usd": "/data/output/rigid_object.usd",
                    "visual_asset": "/data/output/visual_asset.usd",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(run_mesh_to_usd.subprocess, "run", run_isaac_sim)
    monkeypatch.setattr(
        run_mesh_to_usd,
        "run_usd_validation",
        lambda *_args, **_kwargs: {"status": "passed", "passed": True},
    )

    result = run_mesh_to_usd.run_mesh_to_usd(
        str(_asset(tmp_path)),
        str(output),
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
    )
    assert result["output_usd"] == str(rigid_object)
    assert result["visual_asset"] == str(visual_asset)
    assert result["simready_validation"]["passed"] is True


def test_mesh_runner_rejects_stale_generation_report(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / run_mesh_to_usd.GENERATION_REPORT_NAME
    report_path.write_text(
        '{"status": "generated", "output_usd": "/data/output/rigid_object.usd"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_mesh_to_usd.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )

    with pytest.raises(RuntimeError, match="did not create"):
        run_mesh_to_usd.run_mesh_to_usd(
            str(_asset(tmp_path)),
            str(output),
            validate=False,
            accept_eula=True,
            cache_dir=str(tmp_path / "cache"),
        )

    assert not report_path.exists()


def test_mesh_runner_removes_stale_validation_report_when_disabled(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    output.mkdir()
    rigid_object = output / "rigid_object.usd"
    visual_asset = output / "visual_asset.usd"
    generation_report = output / run_mesh_to_usd.GENERATION_REPORT_NAME
    validation_report = output / "simready_validation_report.json"
    validation_report.write_text(
        '{"status": "passed", "passed": true}',
        encoding="utf-8",
    )

    def run_isaac_sim(*_args, **_kwargs):
        assert not validation_report.exists()
        rigid_object.write_text("#usda 1.0\n", encoding="utf-8")
        visual_asset.write_text("#usda 1.0\n", encoding="utf-8")
        generation_report.write_text(
            json.dumps(
                {
                    "status": "generated",
                    "output_usd": "/data/output/rigid_object.usd",
                    "visual_asset": "/data/output/visual_asset.usd",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(run_mesh_to_usd.subprocess, "run", run_isaac_sim)

    result = run_mesh_to_usd.run_mesh_to_usd(
        str(_asset(tmp_path)),
        str(output),
        validate=False,
        accept_eula=True,
        cache_dir=str(tmp_path / "cache"),
    )

    assert not validation_report.exists()
    assert "simready_validation" not in result


def test_validation_command_is_cpu_only_and_offline(tmp_path):
    command = run_validate_usd.build_validation_command(
        str(_asset(tmp_path)),
        str(tmp_path / "output"),
    )

    assert "--network=none" in command
    assert "--gpus" not in command
    assert "ACCEPT_EULA=Y" not in command
    assert run_validate_usd.VALIDATOR_IMAGE_NAME in command


def test_validation_runner_maps_asset_and_preserves_container_path(
    tmp_path,
    monkeypatch,
):
    asset = _asset(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / "simready_validation_report.json"

    def run_validator(*_args, **_kwargs):
        assert not report_path.exists()
        report_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "passed": True,
                    "asset": "/data/input/object.usd",
                    "failed_requirements": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(
        run_validate_usd.subprocess,
        "run",
        run_validator,
    )

    result = run_validate_usd.run_usd_validation(str(asset), str(output))

    assert result["asset"] == str(asset)
    assert result["container_asset"] == "/data/input/object.usd"
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["asset"] == str(asset)


def test_validation_runner_reports_failed_requirements(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / "simready_validation_report.json"

    def run_validator(*_args, **_kwargs):
        report_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "passed": False,
                    "asset": "/data/input/object.usd",
                    "failed_requirements": ["RB.001"],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 1)

    monkeypatch.setattr(
        run_validate_usd.subprocess,
        "run",
        run_validator,
    )

    with pytest.raises(RuntimeError, match="RB.001"):
        run_validate_usd.run_usd_validation(
            str(_asset(tmp_path)),
            str(output),
        )


def test_validation_runner_rejects_missing_fresh_report(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    stale_report = output / "simready_validation_report.json"
    stale_report.write_text('{"status": "passed", "passed": true}', encoding="utf-8")
    monkeypatch.setattr(
        run_validate_usd.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            2,
            stdout="",
            stderr="validator startup failed",
        ),
    )

    with pytest.raises(RuntimeError, match="validator startup failed"):
        run_validate_usd.run_usd_validation(
            str(_asset(tmp_path)),
            str(output),
        )

    assert not stale_report.exists()


def test_validation_profile_excludes_out_of_scope_features():
    specs = Path(__file__).resolve().parents[1] / "validator" / "specs" / "features"
    requirements = set()
    for feature_path in specs.glob("*.json"):
        requirements.update(json.loads(feature_path.read_text())["requirements"])

    assert "COL.001" not in requirements
    assert not any("GRASP" in requirement for requirement in requirements)
    assert {"UN.003", "UN.006", "UN.007", "RB.001", "RB.007"} <= requirements
