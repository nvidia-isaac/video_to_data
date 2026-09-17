"""Tests for the repository-local E2E configuration, CLI, stages, and runner."""

from __future__ import annotations

import ast
import io
import json
import signal
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from tools.e2e import cli
from tools.e2e import config as config_module
from tools.e2e.config import (
    ConfigError,
    build_config,
    load_config,
    set_active,
    update_stage_config,
    write_config,
)
from tools.e2e.import_reconstruction import BundleImportError, import_bundle
from tools.e2e.inspect_reconstruction import inspect_bundle
from tools.e2e.runner import (
    Command,
    Runner,
    Stage,
    StageError,
    _stage_lease,
    discover_checkpoint,
    reconcile_interrupted_stages,
)
from tools.e2e.stages import (
    OPEN_LOOP_EVAL_BOOTSTRAP,
    collect_stages,
    doctor_stages,
    evaluate_stages,
    finetune_stages,
    import_reconstruction_stages,
    inspect_stages,
    pilot_collection_stage,
    read_pilot_measurement,
    reconstruct_stages,
    retarget_stages,
    setup_stages,
    simulate_stages,
    train_expert_stages,
)

TASK_PROFILE = (
    Path(__file__).resolve().parents[3]
    / "robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json"
)


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "source.mp4"
        self.video.write_bytes(b"video")
        self.mano = self.root / "mano"
        (self.mano / "models").mkdir(parents=True)
        (self.mano / "models/MANO_LEFT.pkl").write_bytes(b"left")
        (self.mano / "models/MANO_RIGHT.pkl").write_bytes(b"right")
        self.groot = self.root / "Isaac-GR00T"
        self.groot.mkdir()
        self.checkpoint = self.root / "model_19999.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.bundle = self.make_bundle("bundle")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_bundle(self, name: str, *, marker: bytes = b"bundle") -> Path:
        bundle = self.root / name
        (bundle / "threejs_scene").mkdir(parents=True)
        (bundle / "result.npz").write_bytes(marker + b"-result")
        (bundle / "mesh.obj").write_bytes(marker + b"-mesh")
        (bundle / "manifest.json").write_bytes(marker + b"-manifest")
        (bundle / "threejs_scene/index.html").write_bytes(marker + b"-scene")
        (bundle / "material.png").write_bytes(marker + b"-material")
        return bundle

    def make_config(self, name: str = "run"):
        data = build_config(
            run_root=self.root / name,
            sequence_id="tissue_box_simple",
            embodiment_contract="vega_sharpa_joint",
            task_profile=TASK_PROFILE,
        )
        return write_config(data)

    def make_ready_config(self, name: str = "ready"):
        config = self.make_config(name)
        config = update_stage_config(
            config,
            "setup",
            {
                "mano_dir": self.mano,
                "isaac_groot_dir": self.groot,
                "image_version": "latest",
                "gpu": 0,
            },
            inputs={"mano_dir": self.mano, "isaac_groot_dir": self.groot},
        )
        config = update_stage_config(
            config,
            "reconstruct",
            {
                "video": self.video,
                "object_prompt": "a tissue box",
                "hand_tracking": "hamer",
            },
            inputs={"video": self.video},
            workflow={"object_prompt": "a tissue box", "hand_tracking": "hamer"},
        )
        config = update_stage_config(
            config,
            "import_reconstruction",
            {"bundle": self.bundle},
            inputs={"reconstruction_bundle": self.bundle},
        )
        config = update_stage_config(
            config,
            "retarget",
            {"sequence_id": "tissue_box_simple", "embodiment": "both"},
            workflow={"sequence_id": "tissue_box_simple"},
        )
        config = update_stage_config(
            config,
            "collect",
            {
                "rl_checkpoint": self.checkpoint,
                "sequence_id": "tissue_box_simple",
                "target_successes": 10,
                "pilot_attempts": 32,
                "measured_success_rate": 0.5,
                "measured_success_rate_source": "prior-override",
                "collection_safety_factor": 1.125,
                "frames_per_episode": 20,
                "collection_max_steps": 600,
                "render_envs": 1,
                "reset_arm_noise_rad": 0.015,
                "reset_finger_noise_rad": 0.03,
                "reset_object_xy_noise_m": 0.005,
                "reset_object_yaw_noise_rad": 0.0,
            },
            inputs={"rl_checkpoint": self.checkpoint},
            workflow={
                "target_successes": 10,
                "pilot_attempts": 32,
                "measured_success_rate": 0.5,
                "collection_safety_factor": 1.125,
                "frames_per_episode": 20,
                "collection_max_steps": 600,
                "render_envs": 1,
                "reset_arm_noise_rad": 0.015,
                "reset_finger_noise_rad": 0.03,
                "reset_object_xy_noise_m": 0.005,
                "reset_object_yaw_noise_rad": 0.0,
            },
        )
        return update_stage_config(
            config,
            "finetune",
            {
                "instruction": "lift the box and hold it",
                "epochs": 1.0,
                "global_batch_size": 32,
                "action_horizon": 16,
                "fps": 20,
                "base_model": "nvidia/GR00T-N1.7-3B",
            },
            workflow={
                "epochs": 1.0,
                "global_batch_size": 32,
                "action_horizon": 16,
                "base_model": "nvidia/GR00T-N1.7-3B",
            },
        )


class ConfigTests(Fixture):
    def test_create_and_load_config(self) -> None:
        config = self.make_config()
        loaded = load_config(config.path)
        self.assertEqual(loaded.run_root, self.root / "run")
        self.assertEqual(loaded.host_path("source"), self.root / "run/source")
        self.assertEqual(loaded.container_path("source"), Path("/workspace/e2e/source"))
        self.assertEqual(loaded.inputs, {})
        self.assertEqual(loaded.workflow["state_dim"], 58)
        self.assertEqual(loaded.workflow["action_dim"], 58)
        self.assertEqual(
            loaded.processed_path(),
            self.root / "run/human_motion_data/ego_recon/processed",
        )
        self.assertEqual(
            loaded.processed_path(container=True),
            Path("/workspace/e2e/human_motion_data/ego_recon/processed"),
        )
        loaded.validate_inputs()

    def test_stage_configuration_preserves_contract_owned_values(self) -> None:
        config = self.make_config()
        updated = update_stage_config(
            config,
            "reconstruct",
            {
                "video": self.video,
                "object_prompt": "a tissue box",
                "hand_tracking": "hamer",
            },
            inputs={"video": self.video},
            workflow={"hand_tracking": "hamer"},
        )
        loaded = load_config(updated.path)
        self.assertEqual(loaded.input_path("video"), self.video)
        self.assertEqual(loaded.workflow["object_prompt"], "a tissue box")
        self.assertNotIn("measured_success_rate", loaded.workflow)

    def test_load_rejects_modified_contract_snapshot(self) -> None:
        config = self.make_config("modified-contract")
        snapshot = config.host_path("embodiment_contract")
        value = json.loads(snapshot.read_text())
        value["fps"] = 30
        snapshot.write_text(json.dumps(value))
        with self.assertRaisesRegex(ConfigError, "snapshot differs"):
            load_config(config.path)

    def test_load_rejects_missing_task_profile_snapshot(self) -> None:
        config = self.make_config("missing-profile")
        config.host_path("task_profile").unlink()
        with self.assertRaisesRegex(ConfigError, "missing task profile snapshot"):
            load_config(config.path)

    def test_init_cli_requires_explicit_contracts(self) -> None:
        output = io.StringIO()
        run_root = self.root / "from-cli"
        with redirect_stdout(output):
            result = cli.main(
                [
                    "init",
                    "--run-root",
                    str(run_root),
                    "--sequence-id",
                    "tissue_box_simple",
                    "--embodiment-contract",
                    "vega_sharpa_joint",
                    "--task-profile",
                    str(TASK_PROFILE),
                    "--no-activate",
                ]
            )
        self.assertEqual(result, 0)
        created = load_config(run_root)
        self.assertEqual(created.inputs, {})

    def test_cli_uses_task_profile_for_reconstruction_prompt(self) -> None:
        config = self.make_config("stage-cli")
        with mock.patch.object(cli.Runner, "run_many", return_value=[]):
            result = cli.main(
                [
                    "reconstruct",
                    "--config",
                    str(config.path),
                    "--video",
                    str(self.video),
                ]
            )
        self.assertEqual(result, 0)
        loaded = load_config(config.path)
        self.assertEqual(loaded.stage("reconstruct")["object_prompt"], "a tissue box")
        self.assertEqual(loaded.input_path("video"), self.video)
        self.assertFalse(loaded.stage("reconstruct")["run_gsplat_refinement"])
        self.assertFalse(loaded.stage("reconstruct")["dev"])

    def test_cli_persists_gsplat_and_dev_reconstruction_options(self) -> None:
        config = self.make_config("stage-cli-gsplat")
        with mock.patch.object(cli.Runner, "run_many", return_value=[]):
            result = cli.main(
                [
                    "reconstruct",
                    "--config",
                    str(config.path),
                    "--video",
                    str(self.video),
                    "--hand-tracking",
                    "hamer",
                    "--run-gsplat-refinement",
                    "--dev",
                ]
            )
        self.assertEqual(result, 0)
        parameters = load_config(config.path).stage("reconstruct")
        self.assertTrue(parameters["run_gsplat_refinement"])
        self.assertTrue(parameters["dev"])

    def test_cli_rejects_gsplat_with_dynhamr(self) -> None:
        config = self.make_config("stage-cli-dynhamr-gsplat")
        error = io.StringIO()
        with redirect_stderr(error):
            result = cli.main(
                [
                    "reconstruct",
                    "--config",
                    str(config.path),
                    "--video",
                    str(self.video),
                    "--hand-tracking",
                    "dynhamr",
                    "--run-gsplat-refinement",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn(
            "--run-gsplat-refinement is only supported with --hand-tracking hamer",
            error.getvalue(),
        )

    def test_import_reconstruction_cli_copies_persists_and_resumes(self) -> None:
        config = self.make_config("import-cli")
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli.main(
                [
                    "import-reconstruction",
                    "--config",
                    str(config.path),
                    "--bundle",
                    str(self.bundle),
                ]
            )
        self.assertEqual(result, 0)
        destination = config.host_path("bundle")
        self.assertEqual(
            (destination / "material.png").read_bytes(), b"bundle-material"
        )
        loaded = load_config(config.path)
        self.assertEqual(loaded.input_path("reconstruction_bundle"), self.bundle)
        self.assertEqual(
            loaded.stage("import_reconstruction")["bundle"], str(self.bundle)
        )

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                cli.main(
                    [
                        "import-reconstruction",
                        "--config",
                        str(config.path),
                    ]
                ),
                0,
            )
        self.assertIn("[SKIP] reconstruction.import", output.getvalue())

    def test_import_reconstruction_dry_run_does_not_copy_or_persist(self) -> None:
        config = self.make_config("import-dry-run")
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli.main(
                [
                    "import-reconstruction",
                    "--config",
                    str(config.path),
                    "--bundle",
                    str(self.bundle),
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("[DRY-RUN] reconstruction.import", output.getvalue())
        self.assertFalse(config.host_path("bundle").exists())
        self.assertNotIn(
            "import_reconstruction", load_config(config.path).data["stage_parameters"]
        )

    def test_setup_skip_reconstruction_is_persisted(self) -> None:
        config = self.make_config("setup-skip")
        with mock.patch.object(cli.Runner, "run_many", return_value=[]) as run_many:
            result = cli.main(
                [
                    "setup",
                    "--config",
                    str(config.path),
                    "--mano-dir",
                    str(self.mano),
                    "--isaac-groot-dir",
                    str(self.groot),
                    "--skip-reconstruction",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            [stage.name for stage in run_many.call_args.args[0]],
            ["setup.reconstruction-loader", "setup.robotic-grounding"],
        )
        self.assertTrue(load_config(config.path).stage("setup")["skip_reconstruction"])

        with mock.patch.object(cli.Runner, "run_many", return_value=[]) as run_many:
            result = cli.main(["setup", "--config", str(config.path)])
        self.assertEqual(result, 0)
        self.assertEqual(
            [stage.name for stage in run_many.call_args.args[0]],
            ["setup.reconstruction-loader", "setup.robotic-grounding"],
        )

    def test_setup_keeps_loader_separate_from_full_reconstruction(self) -> None:
        config = self.make_ready_config("setup-full")
        stages = setup_stages(config)
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "setup.reconstruction-host",
                "setup.reconstruction-loader",
                "setup.robotic-grounding",
            ],
        )
        loader_commands = stages[1].commands
        self.assertIn(
            (
                "bash",
                "scripts/build_ego_reconstruction_packages.sh",
                "--mode",
                "hamer",
            ),
            [command.argv for command in stages[0].commands],
        )
        self.assertNotIn(
            "--accept-nvidia-model-eula",
            stages[0].commands[2].argv,
        )
        self.assertTrue(
            any(
                "modules/v2d_task_library_loader/docker/build.py" in command.argv
                for command in loader_commands
            )
        )

    def test_setup_eula_acceptance_is_explicit_and_persisted(self) -> None:
        config = self.make_config("setup-eula")
        with mock.patch.object(cli.Runner, "run_many", return_value=[]) as run_many:
            result = cli.main(
                [
                    "setup",
                    "--config",
                    str(config.path),
                    "--mano-dir",
                    str(self.mano),
                    "--isaac-groot-dir",
                    str(self.groot),
                    "--accept-nvidia-model-eula",
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue(
            load_config(config.path).stage("setup")["accept_nvidia_model_eula"]
        )
        self.assertIn(
            "--accept-nvidia-model-eula",
            run_many.call_args.args[0][0].commands[2].argv,
        )

        with mock.patch.object(cli.Runner, "run_many", return_value=[]) as run_many:
            result = cli.main(["setup", "--config", str(config.path)])
        self.assertEqual(result, 0)
        self.assertIn(
            "--accept-nvidia-model-eula",
            run_many.call_args.args[0][0].commands[2].argv,
        )

    def test_doctor_requires_the_loader_image(self) -> None:
        config = self.make_ready_config("doctor-loader")
        commands = doctor_stages(config)[0].commands
        self.assertTrue(
            any(
                command.argv
                == ("docker", "image", "inspect", "v2d_task_library_loader:latest")
                for command in commands
            )
        )

    def test_inspect_bundle_reports_required_nonempty_files(self) -> None:
        files = inspect_bundle(self.bundle)
        self.assertEqual(
            {path.relative_to(self.bundle).as_posix() for path in files},
            {
                "result.npz",
                "mesh.obj",
                "manifest.json",
                "threejs_scene/index.html",
            },
        )

    def test_import_bundle_is_idempotent_and_rejects_conflicts(self) -> None:
        destination = self.root / "imported"
        self.assertEqual(import_bundle(self.bundle, destination), "imported")
        self.assertEqual(import_bundle(self.bundle, destination), "already-imported")

        other = self.make_bundle("other-bundle", marker=b"different")
        with self.assertRaisesRegex(BundleImportError, "different contents"):
            import_bundle(other, destination)

    def test_import_bundle_requires_complete_nonempty_source(self) -> None:
        incomplete = self.make_bundle("incomplete")
        (incomplete / "mesh.obj").write_bytes(b"")
        with self.assertRaisesRegex(ConfigError, "mesh.obj.*empty"):
            import_bundle(incomplete, self.root / "unused")

    def test_active_run_switching(self) -> None:
        first = self.make_config("first")
        second = self.make_config("second")
        pointer = self.root / ".e2e/current"
        with mock.patch.object(config_module, "ACTIVE_POINTER", pointer):
            set_active(first)
            self.assertEqual(config_module.active_config().path, first.path)
            set_active(second)
            self.assertEqual(config_module.active_config().path, second.path)

    def test_missing_active_config_has_actionable_error(self) -> None:
        pointer = self.root / ".e2e/current"
        with mock.patch.object(config_module, "ACTIVE_POINTER", pointer):
            with self.assertRaisesRegex(ConfigError, "no active E2E run"):
                config_module.active_config()

    def test_input_path_validation(self) -> None:
        self.video.unlink()
        with self.assertRaisesRegex(ConfigError, "input video"):
            update_stage_config(
                self.make_config(),
                "reconstruct",
                {"video": self.video},
                inputs={"video": self.video},
            )

    def test_optional_config_override(self) -> None:
        config = self.make_config()
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli.main(
                ["status", "--config", str(config.path), "--dry-run", "--json"]
            )
        self.assertEqual(result, 0)
        self.assertIn(str(config.path), output.getvalue())


class RunnerTests(Fixture):
    def write_running_manifest(self, config, stage_name: str) -> None:
        manifest = {
            "schema_version": 1,
            "config": str(config.path),
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "stages": {
                stage_name: {
                    "status": "running",
                    "attempts": [
                        {
                            "number": 1,
                            "status": "running",
                            "started_at": "2026-01-01T00:00:00+00:00",
                        }
                    ],
                }
            },
        }
        (config.run_root / "run_manifest.json").write_text(json.dumps(manifest))

    def test_resume_skips_matching_completed_stage(self) -> None:
        config = self.make_config()
        source = self.root / "input.txt"
        source.write_text("input")
        output = config.run_root / "output.txt"
        command = Command(
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
            ),
            self.root,
            "write output",
        )
        stage = Stage("unit.resume", (command,), (source,), (output,))
        runner = Runner(config)
        self.assertEqual(runner.run(stage), "completed")
        self.assertEqual(runner.run(stage), "skipped")

    def test_completed_stage_rejects_input_mismatch(self) -> None:
        config = self.make_config()
        source = self.root / "input.txt"
        source.write_text("first")
        output = config.run_root / "output.txt"
        stage = Stage(
            "unit.mismatch",
            (
                Command(
                    (
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                    ),
                    self.root,
                ),
            ),
            (source,),
            (output,),
        )
        runner = Runner(config)
        runner.run(stage)
        source.write_text("second")
        with self.assertRaisesRegex(StageError, "changed since completion"):
            runner.run(stage)

    def test_completed_stage_rejects_runtime_code_mismatch(self) -> None:
        config = self.make_config("runtime-mismatch")
        output = config.run_root / "output.txt"
        stage = Stage(
            "unit.runtime-mismatch",
            (
                Command(
                    (
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                    ),
                    self.root,
                ),
            ),
            outputs=(output,),
        )
        first = {
            "revision": "first",
            "runtime_tree_sha256": "runtime-one",
            "runtime_file_count": 1,
        }
        second = {
            "revision": "second",
            "runtime_tree_sha256": "runtime-two",
            "runtime_file_count": 1,
        }
        Runner(config, repository=first).run(stage)
        with self.assertRaisesRegex(StageError, "changed since completion"):
            Runner(config, repository=second).run(stage)

    def test_revision_only_change_keeps_matching_runtime_stage_resumable(self) -> None:
        config = self.make_config("revision-only")
        output = config.run_root / "output.txt"
        stage = Stage(
            "unit.revision-only",
            (
                Command(
                    (
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                    ),
                    self.root,
                ),
            ),
            outputs=(output,),
        )
        first = {
            "revision": "first",
            "runtime_tree_sha256": "same-runtime",
            "runtime_file_count": 1,
        }
        second = {
            "revision": "second",
            "runtime_tree_sha256": "same-runtime",
            "runtime_file_count": 1,
        }
        Runner(config, repository=first).run(stage)
        self.assertEqual(Runner(config, repository=second).run(stage), "skipped")
        manifest = json.loads((config.run_root / "run_manifest.json").read_text())
        self.assertEqual(
            manifest["stages"]["unit.revision-only"]["repository"]["revision"],
            "first",
        )

    def test_failed_attempt_is_preserved_on_retry(self) -> None:
        config = self.make_config()
        failed = Stage(
            "unit.failure",
            (Command((sys.executable, "-c", "raise SystemExit(7)"), self.root),),
        )
        runner = Runner(config)
        with self.assertRaises(StageError):
            runner.run(failed)
        successful = Stage(
            "unit.failure",
            (Command((sys.executable, "-c", "print('recovered')"), self.root),),
        )
        runner.run(successful)
        manifest = json.loads((config.run_root / "run_manifest.json").read_text())
        attempts = manifest["stages"]["unit.failure"]["attempts"]
        self.assertEqual(
            [attempt["status"] for attempt in attempts], ["failed", "completed"]
        )
        self.assertEqual(attempts[0]["exit_code"], 7)

    def test_child_segfault_is_recorded_as_failed(self) -> None:
        config = self.make_config("child-segfault")
        stage = Stage(
            "unit.child-segfault",
            (
                Command(
                    (
                        sys.executable,
                        "-c",
                        "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)",
                    ),
                    self.root,
                ),
            ),
        )
        with self.assertRaises(StageError):
            Runner(config).run(stage)

        manifest = json.loads((config.run_root / "run_manifest.json").read_text())
        attempt = manifest["stages"][stage.name]["attempts"][0]
        self.assertEqual(attempt["status"], "failed")
        self.assertEqual(attempt["exit_code"], -signal.SIGSEGV)

    def test_status_keeps_running_stage_when_lease_is_held(self) -> None:
        config = self.make_config("live-lease")
        stage_name = "unit.live"
        self.write_running_manifest(config, stage_name)

        with _stage_lease(config.run_root, stage_name):
            manifest = reconcile_interrupted_stages(config)

        self.assertEqual(manifest["stages"][stage_name]["status"], "running")
        persisted = json.loads((config.run_root / "run_manifest.json").read_text())
        self.assertEqual(persisted["stages"][stage_name]["status"], "running")

    def test_status_persists_interrupted_when_lease_is_gone(self) -> None:
        config = self.make_config("dead-lease")
        stage_name = "unit.dead"
        self.write_running_manifest(config, stage_name)
        output = io.StringIO()

        with redirect_stdout(output):
            result = cli.main(["status", "--config", str(config.path), "--json"])

        self.assertEqual(result, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["stages"][stage_name]["status"], "interrupted")
        persisted = json.loads((config.run_root / "run_manifest.json").read_text())
        stage = persisted["stages"][stage_name]
        self.assertEqual(stage["status"], "interrupted")
        self.assertEqual(stage["attempts"][0]["status"], "interrupted")
        self.assertNotIn("exit_code", stage["attempts"][0])

    def test_retry_preserves_interrupted_attempt(self) -> None:
        config = self.make_config("interrupted-retry")
        stage_name = "unit.interrupted-retry"
        self.write_running_manifest(config, stage_name)
        stage = Stage(
            stage_name,
            (Command((sys.executable, "-c", "print('recovered')"), self.root),),
        )

        self.assertEqual(Runner(config).run(stage), "completed")

        manifest = json.loads((config.run_root / "run_manifest.json").read_text())
        attempts = manifest["stages"][stage_name]["attempts"]
        self.assertEqual(
            [attempt["status"] for attempt in attempts],
            ["interrupted", "completed"],
        )
        self.assertNotIn("exit_code", attempts[0])
        self.assertEqual(attempts[1]["exit_code"], 0)

    def test_checkpoint_discovery_prefers_highest_numbered_complete_checkpoint(
        self,
    ) -> None:
        root = self.root / "finetune"
        for number in (2, 10):
            checkpoint = root / f"checkpoint-{number}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"model")
        self.assertEqual(discover_checkpoint(root).name, "checkpoint-10")


class StageTests(Fixture):
    def test_loaded_vega_tracks_preserve_alignment_and_scale_world_anchors(
        self,
    ) -> None:
        # The full retarget module imports simulator-image dependencies unavailable to
        # host tests. Execute its two pure-numpy functions directly from the AST so this
        # still exercises the production reconciliation code.
        script = (
            config_module.REPO_ROOT
            / "robotic_grounding/scripts/retarget/ego_recon_to_dexmate_sharpa.py"
        )
        tree = ast.parse(script.read_text())
        wanted = {"orthonormalize_rotations", "reconcile_loaded_world_tracks"}
        functions: list[ast.stmt] = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        namespace = {"np": np, "LOADED_WORLD_ALIGNMENT_TOL": 1e-4}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(script), "exec"),
            namespace,
        )
        reconcile = namespace["reconcile_loaded_world_tracks"]

        object_to_world = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
        object_to_world[:, :3, 3] = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        wrist_to_world = object_to_world.copy()
        wrist_to_world[:, :3, 3] = [[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]]
        data = {
            "object_to_world": object_to_world,
            "left_wrist_to_world": wrist_to_world,
        }
        local_joints = np.arange(126, dtype=np.float64).reshape(2, 21, 3) / 1000
        anchors = wrist_to_world[:, :3, 3]
        loader_translation = np.asarray([1.0, -2.0, 3.0])
        loaded_joints = {
            "left": local_joints
            + anchors[:, None, :]
            + loader_translation[None, None, :]
        }
        loaded_object_position = object_to_world[:, :3, 3] + loader_translation[None, :]
        loaded_object_rotation = np.repeat(np.eye(3)[None, :, :], 2, axis=0)

        joints, object_position, object_rotation = reconcile(
            loaded_joints,
            loaded_object_position,
            loaded_object_rotation,
            data,
            2.0,
        )

        np.testing.assert_allclose(
            joints["left"], local_joints + 2.0 * anchors[:, None, :]
        )
        np.testing.assert_allclose(object_position, 2.0 * object_to_world[:, :3, 3])
        np.testing.assert_allclose(object_rotation, loaded_object_rotation)

        rotated = np.repeat(
            np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])[
                None, :, :
            ],
            2,
            axis=0,
        )
        with self.assertRaisesRegex(ValueError, "--no_ground_align"):
            reconcile(
                loaded_joints,
                loaded_object_position,
                rotated,
                data,
                2.0,
            )

    def test_retarget_prepare_passes_the_mounted_sequence_as_dataset_root(self) -> None:
        config = self.make_ready_config()
        prepare = retarget_stages(config, "floating")[0]
        loader_command = prepare.commands[-1].argv
        dataset_root_index = loader_command.index("--dataset_root") + 1
        self.assertEqual(
            Path(loader_command[dataset_root_index]),
            config.host_path("bundle"),
        )
        hmd_index = loader_command.index("--human_motion_data_dir") + 1
        self.assertEqual(
            Path(loader_command[hmd_index]), config.host_path("human_motion_data")
        )
        self.assertIn("--result_subpath", loader_command)
        self.assertIn("--no_ground_align", loader_command)
        self.assertIn("--sequence_name", loader_command)
        floating_command = retarget_stages(config, "floating")[1].commands[0].argv
        output_index = floating_command.index("--output_dir") + 1
        self.assertEqual(
            Path(floating_command[output_index]),
            config.processed_path(container=True),
        )
        vega_command = retarget_stages(config, "vega")[1].commands[0].argv
        output_index = vega_command.index("--output_root") + 1
        self.assertEqual(
            Path(vega_command[output_index]),
            config.processed_path(container=True),
        )
        self.assertIn("--loaded_dir", vega_command)
        self.assertNotIn("--mano_model_dir", vega_command)
        self.assertIn("PYTHONUNBUFFERED=1", vega_command)
        self.assertIn("PYTHONFAULTHANDLER=1", vega_command)

    def test_vega_retarget_publishes_canonical_partition_and_support_surface(
        self,
    ) -> None:
        config = self.make_ready_config()
        stage = retarget_stages(config, "vega")[1]
        self.assertEqual(len(stage.commands), 2)
        self.assertIn(
            config.processed_path()
            / "sequence_id=tissue_box_simple/robot_name=vega_sharpa",
            stage.outputs,
        )
        support = stage.commands[1].argv
        self.assertEqual(support[support.index("--robot_name") + 1], "vega_sharpa")
        self.assertEqual(
            Path(support[support.index("--input_dir") + 1]),
            config.processed_path(container=True),
        )
        self.assertEqual(
            Path(support[support.index("--output") + 1]),
            config.container_path("human_motion_data")
            / "ego_recon/reconstructed_stage/tissue_box_simple_vega_sharpa_support.usda",
        )

    def test_floating_retarget_publishes_verification_video(self) -> None:
        config = self.make_ready_config()
        stage = retarget_stages(config, "floating")[1]
        self.assertEqual(len(stage.commands), 4)

        support_path = (
            config.container_path("human_motion_data")
            / "ego_recon/reconstructed_stage/tissue_box_simple_sharpa_wave_support.usda"
        )
        video = stage.commands[3].argv
        self.assertIn("scripts/retarget/vis_retargeted.py", video)
        self.assertEqual(video[video.index("--robot") + 1], "sharpa_wave")
        self.assertEqual(Path(video[video.index("--support_usd") + 1]), support_path)
        self.assertIn("--save_mp4", video)
        self.assertEqual(
            Path(video[video.index("--mp4_dir") + 1]),
            config.container_path("run_root") / "floating_retarget_qa",
        )
        self.assertIn(
            config.run_root / "floating_retarget_qa" / "tissue_box_simple.mp4",
            stage.outputs,
        )

    def test_reconstruct_forwards_persisted_gsplat_and_dev_options(self) -> None:
        config = update_stage_config(
            self.make_ready_config(),
            "reconstruct",
            {
                "video": self.video,
                "object_prompt": "a tissue box",
                "hand_tracking": "hamer",
                "run_gsplat_refinement": True,
                "dev": True,
            },
            inputs={"video": self.video},
            workflow={"hand_tracking": "hamer"},
        )
        command = reconstruct_stages(config)[0].commands[0].argv
        self.assertIn("--run_gsplat_refinement", command)
        self.assertIn("--dev", command)

    def test_reconstruct_omits_optional_flags_by_default(self) -> None:
        command = reconstruct_stages(self.make_ready_config())[0].commands[0].argv
        self.assertNotIn("--run_gsplat_refinement", command)
        self.assertNotIn("--dev", command)

    def test_every_public_stage_builds_a_dry_run(self) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        builders = {
            "setup": setup_stages(config),
            "doctor": doctor_stages(config),
            "reconstruct": reconstruct_stages(config),
            "import-reconstruction": import_reconstruction_stages(config),
            "inspect": inspect_stages(config),
            "retarget": retarget_stages(config, "both"),
            "simulate": simulate_stages(config, "both"),
            "train-expert": train_expert_stages(
                config, "both", max_iterations=1, num_envs=1
            ),
            "collect-pilot": [pilot_collection_stage(config)],
            "collect": collect_stages(config),
            "finetune": finetune_stages(config, checkpoint=checkpoint_dir),
            "evaluate": evaluate_stages(
                config, checkpoint=checkpoint_dir, episodes=1, num_envs=1
            ),
        }
        for name, stages in builders.items():
            with self.subTest(name=name):
                self.assertTrue(stages)
                output = io.StringIO()
                with redirect_stdout(output):
                    Runner(config, dry_run=True).run_many(stages)
                self.assertIn("[DRY-RUN]", output.getvalue())

    def test_simulate_smoke_is_finite(self) -> None:
        config = self.make_ready_config()
        command = simulate_stages(config, "vega")[0].commands[0].argv
        self.assertEqual(command[command.index("--max_steps") + 1], "2")

    def test_open_loop_uses_standard_joint_evaluator(self) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        command = (
            finetune_stages(config, checkpoint=checkpoint_dir)[-1].commands[0].argv
        )
        self.assertEqual(command[command.index("-c") + 1], OPEN_LOOP_EVAL_BOOTSTRAP)
        self.assertIn("gr00t/eval/open_loop_eval.py", OPEN_LOOP_EVAL_BOOTSTRAP)
        keys = command[
            command.index("--modality-keys") + 1 : command.index("--action-horizon")
        ]
        self.assertEqual(
            keys,
            ("right_arm", "right_finger", "left_arm", "left_finger"),
        )
        self.assertEqual(command[command.index("--action-horizon") + 1], "16")

    def test_open_loop_bootstrap_forces_info_with_an_existing_handler(self) -> None:
        config = self.make_config("open-loop-logging")
        groot = self.root / "fake-groot"
        evaluator = groot / "gr00t/eval/open_loop_eval.py"
        evaluator.parent.mkdir(parents=True)
        (groot / "sitecustomize.py").write_text(
            "import logging\nlogging.getLogger().addHandler(logging.StreamHandler())\n"
        )
        evaluator.write_text(
            """import argparse
import logging
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--save-plot-path', required=True)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logging.info('Dataset length: 10')
logging.info('Average MSE across all trajs: 0.125')
logging.info('Average MAE across all trajs: 0.25')
Path(args.save_plot_path).parent.mkdir(parents=True, exist_ok=True)
Path(args.save_plot_path).write_bytes(b'plot')
"""
        )
        plot = config.run_root / "open_loop/trajectory_0.png"
        stage = Stage(
            "unit.open-loop-logging",
            (
                Command(
                    (
                        sys.executable,
                        "-c",
                        OPEN_LOOP_EVAL_BOOTSTRAP,
                        "--save-plot-path",
                        str(plot),
                    ),
                    groot,
                    env={"PYTHONPATH": str(groot)},
                ),
            ),
            outputs=(plot,),
        )

        self.assertEqual(Runner(config).run(stage), "completed")
        manifest = json.loads((config.run_root / "run_manifest.json").read_text())
        log_path = manifest["stages"][stage.name]["attempts"][0]["log"]
        log = Path(log_path).read_text()
        self.assertIn("Dataset length: 10", log)
        self.assertIn("Average MSE across all trajs: 0.125", log)
        self.assertIn("Average MAE across all trajs: 0.25", log)

    def test_finetune_rejects_a_horizon_outside_the_released_joint_contract(
        self,
    ) -> None:
        config = self.make_ready_config()
        config.data["workflow"]["action_horizon"] = 40
        with self.assertRaisesRegex(ValueError, "action horizon 16"):
            finetune_stages(config)

    def test_evaluate_can_pin_the_repository_motion_and_model_seed(self) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        stage = evaluate_stages(
            config,
            checkpoint=checkpoint_dir,
            episodes=20,
            num_envs=4,
            motion_source="repository",
            model_seed=17,
            execution_length=4,
        )[-1]
        command = stage.commands[0].argv
        self.assertEqual(
            Path(command[command.index("--motion-file") + 1]),
            Path(
                "ego_recon/processed/sequence_id=tissue_box_simple/robot_name=vega_sharpa"
            ),
        )
        self.assertEqual(command[command.index("--model-seed") + 1], "17")
        self.assertEqual(
            command[command.index("--task") + 1],
            "VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0",
        )
        self.assertNotIn("--profile", command)
        self.assertEqual(command[command.index("--execution-length") + 1], "4")
        self.assertIn(
            config_module.REPO_ROOT
            / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/"
            "human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/"
            "robot_name=vega_sharpa",
            stage.inputs,
        )

    def test_dataset_statistics_precede_the_container_dataset_audit(self) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        stages = finetune_stages(config, checkpoint=checkpoint_dir)
        names = [stage.name for stage in stages]
        self.assertLess(
            names.index("finetune.statistics"),
            names.index("finetune.audit-dataset"),
        )
        audit = stages[names.index("finetune.audit-dataset")].commands[0]
        self.assertEqual(audit.cwd, config_module.REPO_ROOT / "robotic_grounding")
        self.assertEqual(
            Path(audit.argv[0]),
            config_module.REPO_ROOT / "robotic_grounding/workflow/run.sh",
        )
        self.assertIn("e2e-run", audit.argv)
        separator = audit.argv.index("--")
        self.assertEqual(
            audit.argv[separator + 1 : separator + 4],
            ("python", "-m", "groot_finetune.tools.audit_groot_run"),
        )
        self.assertNotIn(sys.executable, audit.argv[separator + 1 :])
        self.assertEqual(
            Path(audit.argv[audit.argv.index("--root") + 1]),
            config.container_path("run_root"),
        )
        self.assertEqual(
            Path(audit.argv[audit.argv.index("--contract") + 1]),
            config.container_path("embodiment_contract"),
        )
        self.assertEqual(
            Path(audit.argv[audit.argv.index("--task-profile") + 1]),
            config.container_path("task_profile"),
        )
        self.assertEqual(
            Path(audit.argv[audit.argv.index("--json-output") + 1]),
            config.container_path("run_root") / "dataset_audit.json",
        )

    def test_statistics_inputs_exclude_the_stats_file_the_stage_creates(self) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        statistics = finetune_stages(config, checkpoint=checkpoint_dir)[1]
        self.assertEqual(statistics.name, "finetune.statistics")
        self.assertIn(
            str(config_module.REPO_ROOT / "robotic_grounding"),
            statistics.commands[0].env["PYTHONPATH"],
        )
        self.assertNotIn(config.host_path("dataset"), statistics.inputs)
        self.assertNotIn(
            config.host_path("dataset") / "meta/stats.json",
            statistics.inputs,
        )

    def test_external_groot_modality_commands_receive_repo_pythonpath(self) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        stages = finetune_stages(config, checkpoint=checkpoint_dir)
        repo_package_root = str(config_module.REPO_ROOT / "robotic_grounding")
        for stage_name in (
            "finetune.statistics",
            "finetune.train",
            "finetune.open-loop",
        ):
            stage = next(stage for stage in stages if stage.name == stage_name)
            for command in stage.commands:
                if "uv" in command.argv:
                    self.assertIn(repo_package_root, command.env["PYTHONPATH"])

    def test_dataset_audit_fingerprints_stats_without_the_mutable_dataset_root(
        self,
    ) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        audit = next(
            stage
            for stage in finetune_stages(config, checkpoint=checkpoint_dir)
            if stage.name == "finetune.audit-dataset"
        )
        self.assertNotIn(config.host_path("dataset"), audit.inputs)
        self.assertIn(config.host_path("dataset") / "meta/stats.json", audit.inputs)

    def test_open_loop_declares_standard_output_file(self) -> None:
        config = self.make_ready_config()
        checkpoint_dir = config.host_path("finetune") / "checkpoint-1"
        open_loop = finetune_stages(config, checkpoint=checkpoint_dir)[-1]
        expected = config.host_path("open_loop") / "trajectory_0.png"
        self.assertEqual(open_loop.outputs, (expected,))
        command = open_loop.commands[0].argv
        self.assertIn("gr00t/eval/open_loop_eval.py", command[command.index("-c") + 1])
        self.assertEqual(Path(command[command.index("--save-plot-path") + 1]), expected)
        self.assertNotIn(config.host_path("selected"), open_loop.inputs)

    def test_collection_rollouts_keep_hydra_outputs_in_the_active_run(self) -> None:
        config = self.make_ready_config()
        pilot_command = pilot_collection_stage(config).commands[0].argv
        self.assertIn(
            "hydra.run.dir=/workspace/e2e/source/pilot/hydra",
            pilot_command,
        )
        rollout_command = collect_stages(config)[1].commands[0].argv
        self.assertIn(
            "hydra.run.dir=/workspace/e2e/source/hydra",
            rollout_command,
        )
        self.assertIn("--replace_export", pilot_command)
        self.assertIn("--replace_export", rollout_command)
        self.assertEqual(
            pilot_command[pilot_command.index("--task") + 1],
            "VegaSharpa-WholeBody-Manip-v0",
        )
        self.assertEqual(
            Path(pilot_command[pilot_command.index("--contract") + 1]),
            config.container_path("embodiment_contract"),
        )
        self.assertEqual(
            Path(rollout_command[rollout_command.index("--contract") + 1]),
            config.container_path("embodiment_contract"),
        )
        selection_command = collect_stages(config)[2].commands[0].argv
        self.assertIn("--replace", selection_command)

    def test_collection_plan_does_not_fingerprint_the_mutable_config(self) -> None:
        config = self.make_ready_config()
        plan = collect_stages(config)[0]
        self.assertEqual(plan.inputs, ())

    def test_cli_dry_run_constructs_each_stage_without_artifacts(self) -> None:
        config = self.make_ready_config()
        commands = (
            ["setup"],
            ["doctor"],
            ["reconstruct"],
            ["import-reconstruction"],
            ["inspect"],
            ["retarget", "--embodiment", "both"],
            ["simulate", "--embodiment", "both"],
            ["train-expert", "--embodiment", "both"],
            ["collect"],
            ["finetune"],
            ["evaluate", "--episodes", "1"],
        )
        for command in commands:
            with self.subTest(command=command):
                output = io.StringIO()
                error = io.StringIO()
                with redirect_stdout(output), redirect_stderr(error):
                    result = cli.main(
                        [*command, "--config", str(config.path), "--dry-run"]
                    )
                self.assertEqual(result, 0, error.getvalue())
                self.assertIn("[DRY-RUN]", output.getvalue())

    def test_train_expert_defaults_to_the_documented_vega_smoke(self) -> None:
        config = self.make_ready_config("train-default")
        with mock.patch.object(cli.Runner, "run_many", return_value=[]) as run_many:
            result = cli.main(["train-expert", "--config", str(config.path)])
        self.assertEqual(result, 0)
        stages = run_many.call_args.args[0]
        self.assertEqual([stage.name for stage in stages], ["train-expert.vega"])
        command = stages[0].commands[0].argv
        self.assertIn("--output_root", command)
        self.assertIn("--video", command)
        self.assertIn("--video_length", command)
        self.assertNotIn("--contract", command)
        self.assertNotIn("--task-profile", command)
        self.assertNotIn("--zero-actor", command)
        self.assertFalse(any(arg.startswith("agent.") for arg in command))
        self.assertFalse(any(arg.startswith("env.rewards.") for arg in command))
        interval_index = command.index("--video_interval") + 1
        self.assertEqual(command[interval_index], "8")
        self.assertTrue(any(arg.startswith("hydra.run.dir=") for arg in command))
        configured = load_config(config.path).stage("train_expert")
        self.assertEqual(configured["embodiment"], "vega")
        self.assertEqual(configured["max_iterations"], 1)
        self.assertEqual(configured["num_envs"], 1)

    def test_collect_dry_run_measures_before_planning(self) -> None:
        config = self.make_config("pilot")
        config = update_stage_config(
            config,
            "setup",
            {"mano_dir": self.mano, "isaac_groot_dir": self.groot},
            inputs={"mano_dir": self.mano, "isaac_groot_dir": self.groot},
        )
        config = update_stage_config(
            config,
            "retarget",
            {"sequence_id": "tissue_box_simple"},
            workflow={"sequence_id": "tissue_box_simple"},
        )
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            result = cli.main(
                [
                    "collect",
                    "--config",
                    str(config.path),
                    "--rl-checkpoint",
                    str(self.checkpoint),
                    "--target-successes",
                    "10",
                    "--pilot-attempts",
                    "8",
                    "--collection-max-steps",
                    "600",
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0, error.getvalue())
        self.assertIn("collect.pilot", output.getvalue())
        self.assertIn("waits for the pilot manifest", output.getvalue())
        self.assertNotIn("collect.rollouts", output.getvalue())

    def test_prior_rate_requires_matching_episode_length(self) -> None:
        config = self.make_ready_config("prior")
        error = io.StringIO()
        with redirect_stderr(error):
            result = cli.main(
                [
                    "collect",
                    "--config",
                    str(config.path),
                    "--measured-success-rate",
                    "0.75",
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("--expected-frames is required", error.getvalue())

    def test_changed_measurement_contract_requires_a_new_pilot(self) -> None:
        config = self.make_ready_config("changed-contract")
        replacement = self.root / "replacement.pt"
        replacement.write_bytes(b"new checkpoint")
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            result = cli.main(
                [
                    "collect",
                    "--config",
                    str(config.path),
                    "--rl-checkpoint",
                    str(replacement),
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0, error.getvalue())
        self.assertIn("collect.pilot", output.getvalue())
        self.assertIn("waits for the pilot manifest", output.getvalue())

    def test_pilot_manifest_provides_rate_and_frames(self) -> None:
        manifest = self.root / "pilot-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "episode_count": 8,
                    "source_successful_episode_count": 6,
                    "episode_lengths": [519] * 8,
                }
            )
        )
        self.assertEqual(read_pilot_measurement(manifest), (0.75, 8, 6, 519))

    def test_pilot_uses_success_horizon_when_failures_terminate_early(self) -> None:
        manifest = self.root / "pilot-mixed-lengths.json"
        manifest.write_text(
            json.dumps(
                {
                    "episode_count": 8,
                    "source_successful_episode_count": 7,
                    "episode_lengths": [519, 519, 14, 519, 519, 519, 519, 519],
                    "successful_trajectory_diversity": {"horizon": 519},
                }
            )
        )
        self.assertEqual(read_pilot_measurement(manifest), (0.875, 8, 7, 519))

    def test_zero_success_pilot_stops_full_collection(self) -> None:
        manifest = self.root / "pilot-zero.json"
        manifest.write_text(
            json.dumps(
                {
                    "episode_count": 8,
                    "source_successful_episode_count": 0,
                    "episode_lengths": [519] * 8,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "zero timeout-eligible sources"):
            read_pilot_measurement(manifest)


if __name__ == "__main__":
    unittest.main()
