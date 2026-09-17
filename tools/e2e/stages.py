"""Stage definitions and internal command construction for ``run_e2e.sh``."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import RECONSTRUCTION_BUNDLE_FILES, REPO_ROOT, E2EConfig
from .runner import Command, Stage

RECONSTRUCTION_ROOT = REPO_ROOT / "reconstruction"
ROBOTIC_ROOT = REPO_ROOT / "robotic_grounding"
WORKFLOW_RUNNER = ROBOTIC_ROOT / "workflow" / "run.sh"
CANONICAL_TOOLS = ROBOTIC_ROOT / "groot_finetune" / "tools"
CANONICAL_EVAL = ROBOTIC_ROOT / "groot_finetune" / "closed_loop" / "run_eval.sh"
REPOSITORY_HUMAN_MOTION_DATA = (
    ROBOTIC_ROOT / "source/robotic_grounding/robotic_grounding/assets/human_motion_data"
)
CONTAINER_REPOSITORY_HUMAN_MOTION_DATA = Path(
    "/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data"
)
OPEN_LOOP_EVAL_BOOTSTRAP = (
    "import logging, runpy; "
    "logging.basicConfig(level=logging.INFO, force=True); "
    "runpy.run_path('gr00t/eval/open_loop_eval.py', run_name='__main__')"
)


def _command(
    argv: Iterable[str | Path],
    *,
    cwd: Path,
    label: str,
    env: dict[str, str] | None = None,
) -> Command:
    return Command(tuple(str(value) for value in argv), cwd, label, env or {})


def _container_command(
    config: E2EConfig,
    argv: Iterable[str | Path],
    *,
    label: str,
    workdir: str = "/workspace/video_to_data/robotic_grounding",
) -> Command:
    prefix: list[str | Path] = [
        WORKFLOW_RUNNER,
        "e2e-run",
        str(config.runtime["image_version"]),
        str(config.runtime["gpu"]),
        "--run-data",
        config.run_root,
    ]
    if config.has_input("mano_dir"):
        prefix.extend(("--mano-dir", config.input_path("mano_dir")))
    if config.has_input("rl_checkpoint"):
        prefix.extend(("--checkpoint", config.input_path("rl_checkpoint")))
    prefix.extend(("--workdir", workdir, "--recreate-on-mount-change", "--"))
    return _command((*prefix, *argv), cwd=ROBOTIC_ROOT, label=label)


def _container_env(config: E2EConfig) -> list[str]:
    return [
        "env",
        f"HUMAN_MOTION_DATA_DIR={config.container_path('human_motion_data')}",
        f"ROBOTIC_GROUNDING_INTERMEDIATE_DIR={config.container_path('run_root') / 'intermediate'}",
    ]


def _motion_partition_relative(workflow: Mapping[str, Any], robot_name: str) -> Path:
    """Return the canonical partition path relative to human_motion_data."""
    sequence = str(workflow["sequence_id"])
    return (
        Path("ego_recon/processed")
        / f"sequence_id={sequence}"
        / f"robot_name={robot_name}"
    )


def _motion_partition_host(config: E2EConfig, robot_name: str) -> Path:
    sequence = str(config.workflow["sequence_id"])
    return (
        config.processed_path() / f"sequence_id={sequence}" / f"robot_name={robot_name}"
    )


def _support_surface_relative(workflow: Mapping[str, Any], robot_name: str) -> Path:
    sequence = str(workflow["sequence_id"])
    return (
        Path("ego_recon/reconstructed_stage") / f"{sequence}_{robot_name}_support.usda"
    )


def setup_stages(
    config: E2EConfig,
    *,
    skip_reconstruction: bool = False,
    accept_nvidia_model_eula: bool = False,
) -> list[Stage]:
    """Build setup stages for reconstruction and robotic-grounding runtimes."""
    weight_command = [
        "bash",
        "scripts/download_ego_reconstruction_weights.sh",
        "--mode",
        "hamer_prompt",
    ]
    if accept_nvidia_model_eula:
        weight_command.append("--accept-nvidia-model-eula")
    reconstruction_host = Stage(
        "setup.reconstruction-host",
        (
            _command(
                ("bash", "scripts/install_ego_reconstruction_packages.sh"),
                cwd=RECONSTRUCTION_ROOT,
                label="install reconstruction host wrappers",
            ),
            _command(
                (
                    "bash",
                    "scripts/build_ego_reconstruction_packages.sh",
                    "--mode",
                    "hamer",
                ),
                cwd=RECONSTRUCTION_ROOT,
                label="build HaMeR reconstruction images",
            ),
            _command(
                tuple(weight_command),
                cwd=RECONSTRUCTION_ROOT,
                label="download reconstruction weights",
            ),
        ),
        inputs=(config.input_path("mano_dir"),),
        description="install host wrappers, build reconstruction images, and fetch weights",
    )
    loader = Stage(
        "setup.reconstruction-loader",
        (
            _command(
                (
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    "modules/v2d_task_library_loader/docker",
                ),
                cwd=RECONSTRUCTION_ROOT,
                label="install loader host wrapper",
            ),
            _command(
                (
                    sys.executable,
                    "modules/v2d_task_library_loader/docker/build.py",
                ),
                cwd=RECONSTRUCTION_ROOT,
                label="build reconstruction loader image",
            ),
        ),
        inputs=(config.input_path("mano_dir"),),
        description="install and build the reconstruction loader required by retargeting",
    )
    robotic_grounding = Stage(
        "setup.robotic-grounding",
        (
            _command(
                (
                    WORKFLOW_RUNNER,
                    "build",
                    str(config.runtime["image_version"]),
                    str(config.runtime["gpu"]),
                ),
                cwd=ROBOTIC_ROOT,
                label="build the Robotic Grounding image",
            ),
            _container_command(
                config, ("true",), label="start or reuse the E2E container"
            ),
        ),
        inputs=(config.input_path("mano_dir"),),
        description="build and start the reusable Isaac Lab container",
    )
    if skip_reconstruction:
        return [loader, robotic_grounding]
    return [reconstruction_host, loader, robotic_grounding]


def doctor_stages(config: E2EConfig) -> list[Stage]:
    """Build preflight stages that validate the configured E2E runtimes."""
    groot = config.input_path("isaac_groot_dir")
    checks = (
        _command(("git", "lfs", "fsck"), cwd=REPO_ROOT, label="verify Git LFS objects"),
        _command(("nvidia-smi",), cwd=REPO_ROOT, label="verify NVIDIA GPU"),
        _command(("docker", "version"), cwd=REPO_ROOT, label="verify Docker daemon"),
        _command(
            ("docker", "image", "inspect", "v2d_task_library_loader:latest"),
            cwd=REPO_ROOT,
            label="verify the reconstruction loader image",
        ),
        _command(
            ("uv", "run", "python", "--version"),
            cwd=groot,
            label="verify Isaac-GR00T environment",
        ),
        _container_command(
            config,
            (
                "python",
                "-c",
                "import msgpack_numpy, zmq; print('IsaacLab E2E dependencies OK')",
            ),
            label="verify the mounted Isaac Lab container",
        ),
    )
    return [
        Stage(
            "doctor",
            checks,
            inputs=(
                config.input_path("mano_dir"),
                config.input_path("isaac_groot_dir"),
            ),
            description="validate local tools, inputs, GPU, Docker, mounts, and runtimes",
        )
    ]


def reconstruct_stages(config: E2EConfig) -> list[Stage]:
    """Build stages that reconstruct and validate an input video."""
    output = config.host_path("reconstruction")
    workflow = config.workflow
    parameters = config.stage("reconstruct")
    bundle = config.host_path("bundle")
    argv: list[str | Path] = [
        sys.executable,
        "modules/v2d_pipelines/run_ego_reconstruction.py",
        "--video",
        config.input_path("video"),
        "--object_prompt",
        str(workflow["object_prompt"]),
        "--output_dir",
        output,
        "--reference_frame",
        "0",
        "--undistort",
        "--hand_tracking",
        str(workflow["hand_tracking"]),
        "--run_droid_slam",
        "--run_gravity_alignment",
    ]
    if bool(parameters.get("run_gsplat_refinement", False)):
        argv.append("--run_gsplat_refinement")
    argv.append("--export_threejs_result")
    if bool(parameters.get("dev", False)):
        argv.append("--dev")
    command = _command(
        argv,
        cwd=RECONSTRUCTION_ROOT,
        label="reconstruct the egocentric video",
    )
    return [
        Stage(
            "reconstruct",
            (command,),
            inputs=(config.input_path("video"),),
            outputs=(
                bundle / "result.npz",
                bundle / "mesh.obj",
                bundle / "manifest.json",
                bundle / "threejs_scene/index.html",
            ),
            description="produce the SLAM- and gravity-aligned reconstruction bundle",
        )
    ]


def import_reconstruction_stages(config: E2EConfig) -> list[Stage]:
    """Build the stage that imports a precomputed reconstruction bundle."""
    source = config.input_path("reconstruction_bundle")
    destination = config.host_path("bundle")
    command = _command(
        (
            sys.executable,
            "-m",
            "tools.e2e.import_reconstruction",
            "--source",
            source,
            "--destination",
            destination,
        ),
        cwd=REPO_ROOT,
        label="copy the validated reconstruction bundle into the active run",
    )
    return [
        Stage(
            "reconstruction.import",
            (command,),
            inputs=(source,),
            outputs=tuple(
                destination / relative for relative in RECONSTRUCTION_BUNDLE_FILES
            ),
            description="import a precomputed reconstruction bundle",
        )
    ]


def inspect_stages(config: E2EConfig) -> list[Stage]:
    """Build the stage that inspects an active reconstruction bundle."""
    bundle = config.host_path("bundle")
    return [
        Stage(
            "inspect.reconstruction",
            (
                _command(
                    (
                        sys.executable,
                        "-m",
                        "tools.e2e.inspect_reconstruction",
                        "--bundle",
                        bundle,
                    ),
                    cwd=REPO_ROOT,
                    label="validate and report the reconstruction bundle",
                ),
            ),
            inputs=(bundle,),
            outputs=(
                bundle / "result.npz",
                bundle / "mesh.obj",
                bundle / "manifest.json",
                bundle / "threejs_scene/index.html",
            ),
            description="verify the reconstruction bundle before manual Three.js review",
        )
    ]


def _embodiments(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("floating", "vega")
    if value not in {"floating", "vega"}:
        raise ValueError(f"unsupported embodiment: {value}")
    return (value,)


def retarget_stages(config: E2EConfig, embodiment: str) -> list[Stage]:
    """Build retargeting stages for the selected embodiment."""
    selected = _embodiments(embodiment)
    workflow = config.workflow
    sequence = str(workflow["sequence_id"])
    bundle_host = config.host_path("bundle")
    bundle_container = config.container_path("bundle")
    loaded_host = config.host_path("loaded")
    loaded_container = config.container_path("loaded")
    processed_host = config.processed_path()
    processed_container = config.processed_path(container=True)
    stage_container = (
        config.container_path("human_motion_data") / "ego_recon/reconstructed_stage"
    )
    commands: list[Stage] = []

    if "floating" in selected or "vega" in selected:
        prepare = Stage(
            "retarget.prepare",
            (
                _command(
                    ("mkdir", "-p", loaded_host, processed_host),
                    cwd=REPO_ROOT,
                    label="create run data layout",
                ),
                _command(
                    (
                        sys.executable,
                        "-m",
                        "v2d.task_library_loader.docker.run_loader",
                        "--dataset",
                        "ego_recon",
                        "--output_dir",
                        loaded_host,
                        "--mano_model_dir",
                        config.input_path("mano_dir"),
                        "--human_motion_data_dir",
                        config.host_path("human_motion_data"),
                        "--dataset_root",
                        bundle_host,
                        "--result_subpath",
                        ".",
                        "--no_ground_align",
                        "--sequence_name",
                        sequence,
                        "--object_name",
                        sequence,
                        "--sequence_id",
                        sequence,
                        "--save",
                    ),
                    cwd=RECONSTRUCTION_ROOT,
                    label="load the reconstructed MANO and object data",
                ),
            ),
            inputs=(bundle_host, config.input_path("mano_dir")),
            outputs=(loaded_host,),
            description="prepare and load the ego reconstruction dataset",
        )
        commands.append(prepare)
    if "floating" in selected:
        inner = _container_env(config)
        floating_support_container = (
            stage_container / f"{sequence}_sharpa_wave_support.usda"
        )
        floating_video_dir_container = (
            config.container_path("run_root") / "floating_retarget_qa"
        )
        floating = Stage(
            "retarget.floating",
            (
                _container_command(
                    config,
                    (
                        *inner,
                        "python",
                        "scripts/retarget/ego_recon_to_sharpa.py",
                        "--input_dir",
                        loaded_container,
                        "--output_dir",
                        processed_container,
                        "--sequence_id",
                        sequence,
                        "--save",
                    ),
                    label="retarget to floating-hand Sharpa",
                ),
                _container_command(
                    config,
                    (
                        *inner,
                        "python",
                        "scripts/generate_rigid_urdfs.py",
                        "--dataset",
                        "ego_recon",
                    ),
                    label="generate rigid object assets",
                ),
                _container_command(
                    config,
                    (
                        *inner,
                        "python",
                        "scripts/reconstruct_support_surfaces.py",
                        "--dataset",
                        "ego_recon",
                        "--input_dir",
                        processed_container,
                        "--sequence_id",
                        sequence,
                        "--robot_name",
                        "sharpa_wave",
                        "--output",
                        floating_support_container,
                    ),
                    label="reconstruct support surfaces",
                ),
                _container_command(
                    config,
                    (
                        *inner,
                        "python",
                        "scripts/retarget/vis_retargeted.py",
                        "--input_dir",
                        processed_container,
                        "--robot",
                        "sharpa_wave",
                        "--sequence_id",
                        sequence,
                        "--support_usd",
                        floating_support_container,
                        "--save_mp4",
                        "--mp4_dir",
                        floating_video_dir_container,
                    ),
                    label="render the floating-hand verification video",
                ),
            ),
            inputs=(loaded_host,),
            outputs=(
                _motion_partition_host(config, "sharpa_wave"),
                config.host_path("human_motion_data")
                / _support_surface_relative(workflow, "sharpa_wave"),
                config.run_root / "floating_retarget_qa" / f"{sequence}.mp4",
            ),
            description=(
                "retarget and publish the floating-hand motion, scene, and verification video"
            ),
        )
        commands.append(floating)

    if "vega" in selected:
        vega_args: list[str | Path] = [
            *_container_env(config),
            "PYTHONUNBUFFERED=1",
            "PYTHONFAULTHANDLER=1",
            "python",
            "scripts/retarget/ego_recon_to_dexmate_sharpa.py",
            "--bundle_dir",
            bundle_container,
            "--loaded_dir",
            loaded_container,
            "--sequence_id",
            sequence,
            "--object_name",
            sequence,
            "--output_root",
            processed_container,
            "--artifact_dir",
            config.container_path("run_root") / "vega_retarget_qa",
            "--save",
            "--video",
        ]
        commands.append(
            Stage(
                "retarget.vega",
                (
                    _container_command(
                        config, vega_args, label="retarget to Vega/Dexmate Sharpa"
                    ),
                    _container_command(
                        config,
                        (
                            *_container_env(config),
                            "python",
                            "scripts/reconstruct_support_surfaces.py",
                            "--dataset",
                            "ego_recon",
                            "--input_dir",
                            processed_container,
                            "--sequence_id",
                            sequence,
                            "--robot_name",
                            "vega_sharpa",
                            "--output",
                            stage_container / f"{sequence}_vega_sharpa_support.usda",
                        ),
                        label="reconstruct the Vega-specific support surface",
                    ),
                ),
                inputs=(bundle_host, loaded_host),
                outputs=(
                    _motion_partition_host(config, "vega_sharpa"),
                    config.host_path("human_motion_data")
                    / _support_surface_relative(workflow, "vega_sharpa"),
                    config.run_root / "vega_retarget_qa",
                ),
                description="retarget the reconstruction to the Vega whole-body embodiment",
            )
        )
    return commands


def simulate_stages(config: E2EConfig, embodiment: str) -> list[Stage]:
    """Build simulator smoke-test stages for the selected embodiment."""
    stages: list[Stage] = []
    for name in _embodiments(embodiment):
        task, motion = (
            (
                "Sharpa-V2D-v0-Play",
                _motion_partition_relative(config.workflow, "sharpa_wave"),
            )
            if name == "floating"
            else (
                "VegaSharpa-WholeBody-Manip-v0",
                _motion_partition_relative(config.workflow, "vega_sharpa"),
            )
        )
        command = _container_command(
            config,
            (
                *_container_env(config),
                "python",
                "scripts/rsl_rl/dummy_agent.py",
                "--headless",
                "--task",
                task,
                "--motion_file",
                motion,
                "--num_envs",
                "1",
                "--max_steps",
                "2",
            ),
            label=f"run the {name} Isaac Lab smoke",
        )
        stages.append(
            Stage(
                f"simulate.{name}",
                (command,),
                inputs=(_motion_partition_host(config, motion.name.split("=", 1)[1]),),
                description=f"verify the {name} task registers and advances",
            )
        )
    return stages


def train_expert_stages(
    config: E2EConfig,
    embodiment: str,
    *,
    max_iterations: int,
    num_envs: int,
) -> list[Stage]:
    """Build source-expert training stages for the selected embodiment."""
    stages: list[Stage] = []
    for name in _embodiments(embodiment):
        output_host = config.run_root / "train_expert" / name
        output_container = config.container_path("run_root") / "train_expert" / name
        smoke_video_args: tuple[str, ...] = ()
        if max_iterations == 1 and num_envs == 1:
            smoke_video_args = (
                "--video",
                "--video_length",
                "8",
                "--video_interval",
                "8",
            )
        if name == "floating":
            task, motion = (
                "Sharpa-V2D-v0",
                _motion_partition_relative(config.workflow, "sharpa_wave"),
            )
        else:
            task, motion = (
                "VegaSharpa-WholeBody-Manip-v0",
                _motion_partition_relative(config.workflow, "vega_sharpa"),
            )
        command = _container_command(
            config,
            (
                *_container_env(config),
                "python",
                "scripts/rsl_rl/train.py",
                "--headless",
                "--task",
                task,
                "--motion_file",
                motion,
                "--output_root",
                output_container,
                "--num_envs",
                str(num_envs),
                "--max_iterations",
                str(max_iterations),
                "--logger",
                "tensorboard",
                "--run_name",
                f"{config.workflow['sequence_id']}_{name}_e2e",
                *smoke_video_args,
                f"hydra.run.dir={output_container}/hydra/${{now:%Y-%m-%d_%H-%M-%S}}",
            ),
            label=f"train the {name} source expert",
        )
        stages.append(
            Stage(
                f"train-expert.{name}",
                (command,),
                inputs=(_motion_partition_host(config, motion.name.split("=", 1)[1]),),
                outputs=(output_host,),
                description=f"train the {name} RL source expert",
            )
        )
    return stages


def collection_env_count(config: E2EConfig) -> int:
    """Return the planned number of source-policy collection environments."""
    workflow = config.workflow
    return math.ceil(
        int(workflow["target_successes"])
        / float(workflow["measured_success_rate"])
        * float(workflow["collection_safety_factor"])
    )


def optimizer_step_count(config: E2EConfig) -> int:
    """Return the fine-tuning optimizer steps implied by the run contract."""
    workflow = config.workflow
    frames = int(workflow["frames_per_episode"])
    horizon = int(workflow["action_horizon"])
    if horizon > frames:
        raise ValueError(f"action horizon {horizon} exceeds episode frames {frames}")
    usable = frames - horizon + 1
    return math.ceil(
        float(workflow["epochs"])
        * int(workflow["target_successes"])
        * usable
        / int(workflow["global_batch_size"])
    )


def _rollout_command(
    config: E2EConfig,
    *,
    output: Path,
    num_envs: int,
    seed: int,
    label: str,
) -> Command:
    workflow = config.workflow
    motion_rel = _motion_partition_relative(workflow, "vega_sharpa")
    return _container_command(
        config,
        (
            *_container_env(config),
            "python",
            "scripts/rsl_rl/export_parallel_rollouts.py",
            "--headless",
            "--task",
            str(config.data["contracts"]["embodiment"]["source_task"]),
            "--checkpoint",
            config.container_path("rl_checkpoint"),
            "--motion_file",
            motion_rel,
            "--num_envs",
            str(num_envs),
            "--num_steps",
            str(workflow["collection_max_steps"]),
            "--seed",
            str(seed),
            "--export_dir",
            output,
            "--contract",
            config.container_path("embodiment_contract"),
            "--reset_arm_noise_rad",
            str(workflow["reset_arm_noise_rad"]),
            "--reset_finger_noise_rad",
            str(workflow["reset_finger_noise_rad"]),
            "--reset_object_xy_noise_m",
            str(workflow["reset_object_xy_noise_m"]),
            "--reset_object_yaw_noise_rad",
            str(workflow["reset_object_yaw_noise_rad"]),
            "--replace_export",
            f"hydra.run.dir={output}/hydra",
        ),
        label=label,
    )


def pilot_collection_stage(config: E2EConfig) -> Stage:
    """Build the camera-free pilot used to measure source-policy success."""
    pilot_host = config.host_path("source") / "pilot"
    pilot_container = config.container_path("source") / "pilot"
    processed_host = _motion_partition_host(config, "vega_sharpa")
    attempts = int(config.workflow["pilot_attempts"])
    return Stage(
        "collect.pilot",
        (
            _rollout_command(
                config,
                output=pilot_container,
                num_envs=attempts,
                seed=41,
                label="measure source-policy success on a camera-free pilot",
            ),
        ),
        inputs=(processed_host, config.input_path("rl_checkpoint")),
        outputs=(pilot_host / "manifest.json",),
        description=f"measure timeout eligibility over {attempts} source attempts",
    )


def read_pilot_measurement(path: Path) -> tuple[float, int, int, int]:
    """Return rate, completed, successful, and the uniform pilot frame count."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pilot manifest {path}: {exc}") from exc
    completed = int(manifest.get("episode_count", 0))
    successful = int(manifest.get("source_successful_episode_count", -1))
    if completed <= 0 or successful < 0 or successful > completed:
        raise ValueError(
            f"pilot manifest has invalid counts: completed={completed}, successful={successful}"
        )
    if successful == 0:
        raise ValueError(
            f"pilot completed {completed} attempts with zero timeout-eligible sources; "
            "inspect termination counts before full collection"
        )
    length_values = [int(value) for value in manifest.get("episode_lengths", [])]
    if len(length_values) != completed:
        raise ValueError(
            f"pilot manifest has {len(length_values)} episode lengths for {completed} completed attempts"
        )
    if any(value <= 0 for value in length_values):
        raise ValueError(f"pilot episode lengths must be positive: {length_values}")
    diversity = manifest.get("successful_trajectory_diversity", {})
    frames = int(diversity.get("horizon", 0)) if isinstance(diversity, dict) else 0
    lengths = set(length_values)
    if frames <= 0 and len(lengths) == 1:
        frames = next(iter(lengths))
    if frames <= 0:
        raise ValueError(
            f"pilot manifest does not declare the successful episode horizon; attempt lengths were {sorted(lengths)}"
        )
    return successful / completed, completed, successful, frames


def collect_stages(
    config: E2EConfig,
    *,
    extra_sources: Sequence[Path] = (),
    render_envs: int | None = None,
) -> list[Stage]:
    """Build source rollout, selection, recording, and audit stages."""
    workflow = config.workflow
    target = int(workflow["target_successes"])
    frames = int(workflow["frames_per_episode"])
    collection_envs = collection_env_count(config)
    motion_rel = _motion_partition_relative(workflow, "vega_sharpa")
    processed_host = _motion_partition_host(config, "vega_sharpa")
    source_host = config.host_path("source")
    selected_host = config.host_path("selected")
    recording_host = config.host_path("recording")
    source_container = config.container_path("source")
    selected_container = config.container_path("selected")
    recording_container = config.container_path("recording")
    container_env = _container_env(config)
    plan_command = _container_command(
        config,
        (
            "python",
            "-m",
            "groot_finetune.tools.plan_groot_run",
            "--target-successes",
            str(target),
            "--measured-success-rate",
            str(workflow["measured_success_rate"]),
            "--collection-safety-factor",
            str(workflow["collection_safety_factor"]),
        ),
        label="calculate camera-free collection quantity",
    )
    rollout_command = _rollout_command(
        config,
        output=source_container,
        num_envs=collection_envs,
        seed=42,
        label="collect camera-free source rollouts",
    )
    sources_host = (source_host, *extra_sources)
    sources_container = [source_container]
    for source in extra_sources:
        try:
            relative = source.resolve().relative_to(config.run_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"extra source must be inside the active run root: {source}"
            ) from exc
        sources_container.append(config.container_path("run_root") / relative)
    selector_args: list[str | Path] = [
        "python",
        "-m",
        "groot_finetune.tools.select_successful_episodes",
    ]
    for source in sources_container:
        selector_args.extend(("--input", source))
    selector_args.extend(
        (
            "--output",
            selected_container,
            "--target",
            str(target),
            "--expected-frames",
            str(frames),
            "--replace",
        )
    )
    replay_command = _container_command(
        config,
        (
            *container_env,
            "python",
            "scripts/rsl_rl/replay_record.py",
            "--headless",
            "--export_dir",
            selected_container,
            "--motion_file",
            motion_rel,
            "--contract",
            config.container_path("embodiment_contract"),
            "--task_profile",
            config.container_path("task_profile"),
            "--num_envs",
            str(render_envs or workflow["render_envs"]),
            "--num_demos",
            str(target),
            "--save_on",
            "source_success",
            "--record_output",
            recording_container,
            "--verify",
        ),
        label="replay selected episodes with cameras",
    )
    audit_command = _container_command(
        config,
        (
            "python",
            "-m",
            "groot_finetune.tools.audit_groot_run",
            "--root",
            config.container_path("run_root"),
            "--episodes",
            str(target),
            "--frames",
            str(frames),
            "--contract",
            config.container_path("embodiment_contract"),
            "--task-profile",
            config.container_path("task_profile"),
            "--selected-export",
            "selected",
            "--hdf5",
            "recording/data.h5",
            "--json-output",
            config.container_path("run_root") / "semantic_audit.json",
        ),
        label="audit selected and semantic artifacts",
    )
    return [
        Stage(
            "collect.plan",
            (plan_command,),
            description="calculate collection attempts from the measured success rate",
        ),
        Stage(
            "collect.rollouts",
            (rollout_command,),
            inputs=(processed_host, config.input_path("rl_checkpoint")),
            outputs=(source_host / "manifest.json",),
            description="collect bounded Vega source-policy attempts",
        ),
        Stage(
            "collect.select",
            (
                _container_command(
                    config, selector_args, label="select exact successful episodes"
                ),
            ),
            inputs=tuple(sources_host),
            outputs=(selected_host / "manifest.json",),
            description="validate and select the requested number of successes",
        ),
        Stage(
            "collect.replay",
            (replay_command,),
            inputs=(selected_host, processed_host),
            outputs=(recording_host / "data.h5",),
            description="create marker-free semantic recordings with fixed per-demo visuals",
        ),
        Stage(
            "collect.audit",
            (audit_command,),
            inputs=(selected_host, recording_host / "data.h5"),
            outputs=(config.run_root / "semantic_audit.json",),
            description="enforce exact selection and HDF5 contracts",
        ),
    ]


def finetune_stages(
    config: E2EConfig,
    *,
    checkpoint: Path | None = None,
    include_open_loop: bool = True,
) -> list[Stage]:
    """Build dataset conversion, fine-tuning, and open-loop stages."""
    workflow = config.workflow
    configured_horizon = int(workflow["action_horizon"])
    contract_horizon = int(config.data["contracts"]["embodiment"]["action_horizon"])
    if configured_horizon != contract_horizon:
        raise ValueError(
            f"The embodiment contract uses action horizon {contract_horizon}; got {configured_horizon}."
        )
    target = int(workflow["target_successes"])
    frames = int(workflow["frames_per_episode"])
    optimizer_steps = optimizer_step_count(config)
    recording_host = config.host_path("recording") / "data.h5"
    dataset_host = config.host_path("dataset")
    dataset_container = config.container_path("dataset")
    groot = config.input_path("isaac_groot_dir")
    modality = (
        ROBOTIC_ROOT / str(config.data["contracts"]["embodiment"]["modality_config"])
    ).resolve()
    if not modality.is_relative_to(ROBOTIC_ROOT):
        raise ValueError("embodiment modality config must stay under robotic_grounding")
    if not modality.is_file():
        raise FileNotFoundError(
            f"embodiment modality config does not exist: {modality}"
        )
    convert = _container_command(
        config,
        (
            "python",
            "-m",
            "groot_finetune.convert_to_gr00t",
            "--input",
            config.container_path("recording") / "data.h5",
            "--output",
            dataset_container,
            "--contract",
            config.container_path("embodiment_contract"),
            "--task-profile",
            config.container_path("task_profile"),
        ),
        label="convert semantic HDF5 to LeRobot",
    )
    audit = _container_command(
        config,
        (
            "python",
            "-m",
            "groot_finetune.tools.audit_groot_run",
            "--root",
            config.container_path("run_root"),
            "--episodes",
            str(target),
            "--frames",
            str(frames),
            "--contract",
            config.container_path("embodiment_contract"),
            "--task-profile",
            config.container_path("task_profile"),
            "--selected-export",
            "selected",
            "--hdf5",
            "recording/data.h5",
            "--dataset",
            "gr00t_dataset",
            "--json-output",
            config.container_path("run_root") / "dataset_audit.json",
        ),
        label="audit the complete dataset chain",
    )
    statistics = _command(
        (
            "uv",
            "run",
            "python",
            "gr00t/data/stats.py",
            "--dataset-path",
            dataset_host,
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            modality,
        ),
        cwd=groot,
        label="generate GR00T dataset statistics",
        env={"PYTHONPATH": f"{groot}:{ROBOTIC_ROOT}"},
    )
    train = _command(
        (
            "uv",
            "run",
            "python",
            "gr00t/experiment/launch_finetune.py",
            "--base-model-path",
            str(workflow["base_model"]),
            "--dataset-path",
            dataset_host,
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            modality,
            "--num-gpus",
            "1",
            "--output-dir",
            config.host_path("finetune"),
            "--max-steps",
            str(optimizer_steps),
            "--global-batch-size",
            str(workflow["global_batch_size"]),
            "--save-steps",
            str(optimizer_steps),
        ),
        cwd=groot,
        label="fine-tune GR00T N1.7",
        env={"PYTHONPATH": f"{groot}:{ROBOTIC_ROOT}"},
    )
    stages = [
        Stage(
            "finetune.convert",
            (convert,),
            inputs=(recording_host,),
            outputs=(dataset_host / "meta/info.json",),
            description="convert recordings to the GR00T LeRobot contract",
        ),
        Stage(
            "finetune.statistics",
            (statistics,),
            inputs=(
                dataset_host / "meta/info.json",
                dataset_host / "meta/modality.json",
                dataset_host / "meta/episodes.jsonl",
                dataset_host / "data",
                modality,
            ),
            outputs=(dataset_host / "meta/stats.json",),
            description=(
                f"generate statistics for the {config.data['contracts']['embodiment']['contract_id']} contract"
            ),
        ),
        Stage(
            "finetune.audit-dataset",
            (audit,),
            inputs=(
                dataset_host / "meta/info.json",
                dataset_host / "meta/modality.json",
                dataset_host / "meta/stats.json",
                dataset_host / "data",
                dataset_host / "videos",
                recording_host,
                config.host_path("selected"),
            ),
            outputs=(config.run_root / "dataset_audit.json",),
            description="prove exact row, frame, video, state, and action counts",
        ),
        Stage(
            "finetune.train",
            (train,),
            inputs=(dataset_host / "meta/stats.json", modality),
            outputs=(config.host_path("finetune"),),
            description=f"train for {optimizer_steps} optimizer steps",
        ),
    ]
    if include_open_loop:
        resolved = checkpoint or (
            config.host_path("finetune") / f"checkpoint-{optimizer_steps}"
        )
        open_loop_plot = config.host_path("open_loop") / "trajectory_0.png"
        action_keys = [
            str(field["key"])
            for field in config.data["contracts"]["embodiment"]["action_fields"]
        ]
        open_loop = _command(
            (
                "uv",
                "run",
                "python",
                "-c",
                OPEN_LOOP_EVAL_BOOTSTRAP,
                "--dataset-path",
                dataset_host,
                "--embodiment-tag",
                "NEW_EMBODIMENT",
                "--model-path",
                resolved,
                "--modality-keys",
                *action_keys,
                "--action-horizon",
                str(contract_horizon),
                "--save-plot-path",
                open_loop_plot,
            ),
            cwd=groot,
            env={"PYTHONPATH": f"{groot}:{ROBOTIC_ROOT}"},
            label="run Isaac-GR00T open-loop evaluation for every action key",
        )
        stages.append(
            Stage(
                "finetune.open-loop",
                (open_loop,),
                inputs=(dataset_host, resolved),
                outputs=(open_loop_plot,),
                description="measure finite open-loop MSE/MAE for every action key",
            )
        )
    return stages


def evaluate_stages(
    config: E2EConfig,
    *,
    checkpoint: Path,
    episodes: int,
    num_envs: int,
    motion_source: str = "run",
    model_seed: int = 0,
    execution_length: int = 4,
    evaluation_horizon: int | None = None,
) -> list[Stage]:
    """Build a closed-loop evaluation stage for a fine-tuned checkpoint."""
    workflow = config.workflow
    embodiment_contract = config.data["contracts"]["embodiment"]
    horizon = int(
        evaluation_horizon
        if evaluation_horizon is not None
        else workflow["frames_per_episode"]
    )
    if motion_source == "run":
        human_motion_host = config.host_path("human_motion_data")
        human_motion_container = config.container_path("human_motion_data")
    elif motion_source == "repository":
        human_motion_host = REPOSITORY_HUMAN_MOTION_DATA
        human_motion_container = CONTAINER_REPOSITORY_HUMAN_MOTION_DATA
    else:
        raise ValueError(f"unsupported evaluation motion source: {motion_source}")
    motion_rel = _motion_partition_relative(workflow, "vega_sharpa")
    motion_host = human_motion_host / motion_rel
    support_host = human_motion_host / _support_surface_relative(
        workflow, "vega_sharpa"
    )
    output_host = config.host_path("closed_loop") / f"evaluation_{episodes}.json"
    output_container = (
        config.container_path("closed_loop") / f"evaluation_{episodes}.json"
    )
    videos_container = config.container_path("closed_loop") / f"success_{episodes}"
    ensure = _container_command(
        config, ("true",), label="start or reuse the E2E container"
    )
    evaluate = _command(
        (
            "bash",
            CANONICAL_EVAL,
            "--gr00t-dir",
            config.input_path("isaac_groot_dir"),
            "--model",
            checkpoint,
            "--container",
            str(config.runtime["container_name"]),
            "--expected-mount-source",
            REPO_ROOT,
            "--client-workdir",
            "/workspace/video_to_data/robotic_grounding",
            "--task",
            str(embodiment_contract["inference_task"]),
            "--contract",
            config.container_path("embodiment_contract"),
            "--task-profile",
            config.container_path("task_profile"),
            "--motion-file",
            motion_rel,
            "--human-motion-data-dir",
            human_motion_container,
            "--expected-sequence-id",
            str(workflow["sequence_id"]),
            "--expected-robot-name",
            str(embodiment_contract["robot_name"]),
            "--output-json",
            output_container,
            "--episodes",
            str(episodes),
            "--num-envs",
            str(num_envs),
            "--episode-horizon",
            str(horizon),
            "--execution-length",
            str(execution_length),
            "--model-seed",
            str(model_seed),
            "--success-video-dir",
            videos_container,
            "--success-video-camera",
            str(embodiment_contract["cameras"][0]["key"]),
            "--max-success-videos",
            "1" if episodes == 1 else "3",
            "--server-log",
            config.run_root / "logs" / f"groot-server-{episodes}.log",
            # The client resolves its task config through Hydra, whose default output
            # directory is relative to the in-container working directory and is not
            # writable there. Route it into the run root, as the collection commands do.
            "--client-extra-arg",
            f"hydra.run.dir={config.container_path('closed_loop')}/hydra_{episodes}",
            "--client-extra-arg",
            "hydra.output_subdir=null",
        ),
        cwd=REPO_ROOT,
        label="run the managed GR00T server and Isaac Lab client",
    )
    return [
        Stage(
            "evaluate.container",
            (ensure,),
            inputs=(config.input_path("mano_dir"),),
            description="ensure the exact configured container and mounts are active",
        ),
        Stage(
            f"evaluate.closed-loop-{episodes}",
            (evaluate,),
            inputs=(checkpoint, motion_host, support_host),
            outputs=(output_host,),
            description=f"run {episodes} task-level closed-loop episode(s)",
        ),
    ]
