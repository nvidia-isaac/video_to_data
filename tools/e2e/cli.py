"""Command-line interface for the local reconstruction-to-GR00T workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import (
    ACTIVE_POINTER,
    ConfigError,
    E2EConfig,
    build_config,
    load_config,
    resolve_config,
    set_active,
    update_stage_config,
    write_config,
)
from .runner import (
    Runner,
    StageError,
    discover_checkpoint,
    reconcile_interrupted_stages,
)
from .stages import (
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _normalize_global_args(argv: Sequence[str]) -> list[str]:
    """Allow runner-wide options before or after a subcommand."""
    value_options = {"--config"}
    flag_options = {"--dry-run", "--force", "--no-resume"}
    globals_: list[str] = []
    remainder: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in value_options:
            if index + 1 >= len(argv):
                globals_.append(token)
                index += 1
            else:
                globals_.extend((token, argv[index + 1]))
                index += 2
        elif any(token.startswith(option + "=") for option in value_options):
            globals_.append(token)
            index += 1
        elif token in flag_options:
            globals_.append(token)
            index += 1
        else:
            remainder.append(token)
            index += 1
    return [*globals_, *remainder]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the E2E workflow."""
    parser = argparse.ArgumentParser(
        prog="run_e2e.sh",
        description="Run the local video-to-data reconstruction and GR00T workflow.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="one-command override; defaults to the active .e2e/current run",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace config or rerun completed stages"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun rather than skip matching completed stages",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create and activate an empty run")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--sequence-id", required=True)
    init.add_argument("--embodiment-contract", required=True)
    init.add_argument("--task-profile", type=Path, required=True)
    init.add_argument("--no-activate", action="store_true")

    use = subparsers.add_parser("use", help="switch the active run")
    use.add_argument("path", type=Path, help="run directory or e2e_config.json")

    setup = subparsers.add_parser(
        "setup", help="install dependencies and build/start containers"
    )
    setup.add_argument("--mano-dir", type=Path)
    setup.add_argument("--isaac-groot-dir", type=Path)
    setup.add_argument("--image-version")
    setup.add_argument("--gpu", type=_nonnegative_int)
    setup.add_argument(
        "--skip-reconstruction",
        action="store_true",
        default=None,
        help="skip reconstruction package, image, and weight setup",
    )
    setup.add_argument(
        "--accept-nvidia-model-eula",
        action="store_true",
        default=None,
        help="accept the NVIDIA Open Model License for FoundationPose weights",
    )

    subparsers.add_parser(
        "doctor", help="check configured tools, runtimes, GPU, and mounts"
    )

    reconstruct = subparsers.add_parser(
        "reconstruct", help="reconstruct a source video"
    )
    reconstruct.add_argument("--video", type=Path)
    reconstruct.add_argument("--hand-tracking", choices=("hamer", "dynhamr"))
    reconstruct.add_argument(
        "--run-gsplat-refinement",
        action="store_true",
        default=None,
        help="refine the HaMeR reconstruction with gsplat",
    )
    reconstruct.add_argument(
        "--dev",
        action="store_true",
        default=None,
        help="mount local reconstruction module sources in the worker containers",
    )

    import_reconstruction = subparsers.add_parser(
        "import-reconstruction", help="copy a precomputed reconstruction bundle"
    )
    import_reconstruction.add_argument("--bundle", type=Path)

    subparsers.add_parser(
        "inspect", help="verify the reconstruction bundle for manual review"
    )

    retarget = subparsers.add_parser(
        "retarget", help="retarget the reconstructed motion"
    )
    retarget.add_argument("--embodiment", choices=("floating", "vega", "both"))

    simulate = subparsers.add_parser("simulate", help="run Isaac Lab task smoke checks")
    simulate.add_argument("--embodiment", choices=("floating", "vega", "both"))

    train = subparsers.add_parser("train-expert", help="train source RL experts")
    train.add_argument("--embodiment", choices=("floating", "vega", "both"))
    train.add_argument("--max-iterations", type=_positive_int)
    train.add_argument("--num-envs", type=_positive_int)

    collect = subparsers.add_parser(
        "collect", help="measure, collect, select, replay, and audit demonstrations"
    )
    collect.add_argument("--rl-checkpoint", type=Path)
    collect.add_argument("--target-successes", type=_positive_int)
    collect.add_argument("--pilot-attempts", type=_positive_int)
    collect.add_argument(
        "--measured-success-rate",
        type=_probability,
        help="reuse an equivalent prior measurement instead of running a pilot",
    )
    collect.add_argument(
        "--expected-frames",
        type=_positive_int,
        help="episode length paired with an explicit prior success rate",
    )
    collect.add_argument("--collection-safety-factor", type=_positive_float)
    collect.add_argument("--collection-max-steps", type=_positive_int)
    collect.add_argument("--render-envs", type=_positive_int)
    collect.add_argument("--reset-arm-noise-rad", type=_nonnegative_float)
    collect.add_argument("--reset-finger-noise-rad", type=_nonnegative_float)
    collect.add_argument("--reset-object-xy-noise-m", type=_nonnegative_float)
    collect.add_argument("--reset-object-yaw-noise-rad", type=_nonnegative_float)
    collect.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="additional source export inside RUN_ROOT; repeat for multiple batches",
    )

    finetune = subparsers.add_parser(
        "finetune", help="convert, audit, train, and open-loop evaluate GR00T"
    )
    finetune.add_argument("--epochs", type=_positive_float)
    finetune.add_argument("--global-batch-size", type=_positive_int)
    finetune.add_argument("--base-model")
    finetune.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint to use for open-loop evaluation (training still runs)",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="run managed closed-loop GR00T evaluation"
    )
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--episodes", type=_positive_int)
    evaluate.add_argument("--num-envs", type=_positive_int)
    evaluate.add_argument("--evaluation-horizon", type=_positive_int)
    evaluate.add_argument(
        "--execution-length",
        type=_positive_int,
        help=(
            "actions consumed per predicted 16-step chunk before re-querying (default: 4)"
        ),
    )
    evaluate.add_argument(
        "--motion-source",
        choices=("run", "repository"),
        help=(
            "motion/support artifacts to evaluate: the active run or the repository asset partition (default: run)"
        ),
    )
    evaluate.add_argument(
        "--model-seed",
        type=_nonnegative_int,
        help="seed for GR00T diffusion action sampling (default: 0)",
    )

    status = subparsers.add_parser(
        "status", help="show the active config and stage manifest"
    )
    status.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _runner(args: argparse.Namespace, config: E2EConfig) -> Runner:
    return Runner(
        config,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        force=args.force,
    )


def _stage_value(
    args: argparse.Namespace,
    config: E2EConfig,
    stage: str,
    name: str,
    *,
    default: Any = None,
    required: bool = False,
) -> Any:
    value = getattr(args, name, None)
    if value is None:
        value = config.stage(stage).get(name, default)
    if required and value is None:
        option = "--" + name.replace("_", "-")
        raise ConfigError(
            f"{option} is required the first time this stage is configured"
        )
    return value


def _input_value(
    args: argparse.Namespace,
    config: E2EConfig,
    stage: str,
    name: str,
) -> Path:
    value = _stage_value(args, config, stage, name)
    if value is None and config.has_input(name):
        value = config.input_path(name)
    if value is None:
        option = "--" + name.replace("_", "-")
        raise ConfigError(
            f"{option} is required the first time this stage is configured"
        )
    return Path(value)


def _configure(
    args: argparse.Namespace,
    config: E2EConfig,
    stage: str,
    parameters: dict[str, Any],
    *,
    inputs: dict[str, Path] | None = None,
    runtime: dict[str, Any] | None = None,
    workflow: dict[str, Any] | None = None,
) -> E2EConfig:
    return update_stage_config(
        config,
        stage,
        parameters,
        inputs=inputs,
        runtime=runtime,
        workflow=workflow,
        persist=not args.dry_run,
    )


def _status(config: E2EConfig, *, as_json: bool) -> None:
    manifest_path = config.run_root / "run_manifest.json"
    if manifest_path.is_file():
        manifest = reconcile_interrupted_stages(config)
    else:
        manifest = {"schema_version": 1, "stages": {}}
    stages = manifest.get("stages", {})
    configured = sorted(config.data.get("stage_parameters", {}))
    summary: dict[str, Any] = {
        "active_pointer": str(ACTIVE_POINTER),
        "config": str(config.path),
        "run_root": str(config.run_root),
        "manifest": str(manifest_path),
        "configured_stages": configured,
        "stages": {
            name: {
                "status": value.get("status", "unknown"),
                "completed_at": value.get("completed_at"),
                "last_error": value.get("last_error"),
            }
            for name, value in sorted(stages.items())
        },
    }
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"Config:     {config.path}")
    print(f"Run root:   {config.run_root}")
    print(f"Configured: {', '.join(configured) if configured else 'none'}")
    print(f"Manifest:   {manifest_path if manifest_path.exists() else '(not created)'}")
    if not summary["stages"]:
        print("Stages:     none")
        return
    print("Stages:")
    for name, value in summary["stages"].items():
        suffix = f" — {value['last_error']}" if value["last_error"] else ""
        print(f"  {name:<32} {value['status']}{suffix}")


def _run_collect(args: argparse.Namespace, config: E2EConfig) -> None:
    stage = "collect"
    sequence_id = config.workflow.get("sequence_id")
    if not isinstance(sequence_id, str):
        raise ConfigError(
            "sequence ID is not configured; run retarget with --sequence-id"
        )
    checkpoint = _input_value(args, config, stage, "rl_checkpoint")
    target = int(_stage_value(args, config, stage, "target_successes", default=10))
    pilot_attempts = int(
        _stage_value(args, config, stage, "pilot_attempts", default=32)
    )
    safety = float(
        _stage_value(args, config, stage, "collection_safety_factor", default=1.125)
    )
    collection_max_steps = int(
        _stage_value(args, config, stage, "collection_max_steps", required=True)
    )
    render_envs = int(_stage_value(args, config, stage, "render_envs", default=1))
    arm_noise = float(
        _stage_value(args, config, stage, "reset_arm_noise_rad", default=0.0)
    )
    finger_noise = float(
        _stage_value(args, config, stage, "reset_finger_noise_rad", default=0.0)
    )
    object_xy_noise = float(
        _stage_value(args, config, stage, "reset_object_xy_noise_m", default=0.0)
    )
    object_yaw_noise = float(
        _stage_value(args, config, stage, "reset_object_yaw_noise_rad", default=0.0)
    )
    existing = config.stage(stage)
    rate = args.measured_success_rate
    frames = args.expected_frames
    rate_source: str | None = None
    if rate is not None:
        rate_source = "prior-override"
        if frames is None:
            raise ConfigError(
                "--expected-frames is required with --measured-success-rate; "
                "both values must come from the same equivalent prior run"
            )
    measurement_contract_matches = all(
        (
            str(Path(str(existing.get("rl_checkpoint", ""))).resolve())
            == str(checkpoint.resolve())
            if key == "rl_checkpoint"
            else existing.get(key) == value
        )
        for key, value in {
            "rl_checkpoint": checkpoint,
            "sequence_id": sequence_id,
            "pilot_attempts": pilot_attempts,
            "collection_max_steps": collection_max_steps,
            "reset_arm_noise_rad": arm_noise,
            "reset_finger_noise_rad": finger_noise,
            "reset_object_xy_noise_m": object_xy_noise,
            "reset_object_yaw_noise_rad": object_yaw_noise,
        }.items()
    )
    if (
        rate is None
        and existing.get("measured_success_rate") is not None
        and existing.get("measured_success_rate_source") != "pilot"
        and measurement_contract_matches
    ):
        rate = float(existing["measured_success_rate"])
        frames = int(existing["frames_per_episode"])
        rate_source = str(existing.get("measured_success_rate_source", "configured"))

    parameters: dict[str, Any] = {
        "rl_checkpoint": checkpoint,
        "sequence_id": sequence_id,
        "target_successes": target,
        "pilot_attempts": pilot_attempts,
        "collection_safety_factor": safety,
        "collection_max_steps": collection_max_steps,
        "render_envs": render_envs,
        "reset_arm_noise_rad": arm_noise,
        "reset_finger_noise_rad": finger_noise,
        "reset_object_xy_noise_m": object_xy_noise,
        "reset_object_yaw_noise_rad": object_yaw_noise,
    }
    workflow: dict[str, Any] = {
        key: value for key, value in parameters.items() if key != "rl_checkpoint"
    }
    workflow.update(measured_success_rate=None, frames_per_episode=None)
    if rate is not None and frames is not None:
        parameters.update(
            measured_success_rate=rate,
            measured_success_rate_source=rate_source,
            frames_per_episode=frames,
        )
        workflow.update(
            measured_success_rate=rate,
            frames_per_episode=frames,
        )
    config = _configure(
        args,
        config,
        stage,
        parameters,
        inputs={"rl_checkpoint": checkpoint},
        workflow=workflow,
    )

    pilot = config.host_path("source") / "pilot"
    pilot_manifest = pilot / "manifest.json"
    if rate is None:
        if args.dry_run and not pilot_manifest.is_file():
            _runner(args, config).run(pilot_collection_stage(config))
            print("[DRY-RUN] full collection planning waits for the pilot manifest")
            return
        _runner(args, config).run(pilot_collection_stage(config))
        rate, completed, successful, frames = read_pilot_measurement(pilot_manifest)
        parameters.update(
            measured_success_rate=rate,
            measured_success_rate_source="pilot",
            pilot_completed_attempts=completed,
            pilot_successful_attempts=successful,
            frames_per_episode=frames,
        )
        workflow.update(
            measured_success_rate=rate,
            frames_per_episode=frames,
        )
        config = _configure(
            args,
            config,
            stage,
            parameters,
            inputs={"rl_checkpoint": checkpoint},
            workflow=workflow,
        )
        print(
            f"Measured timeout-eligible sources: {successful}/{completed} = {rate:.4f}; episode frames: {frames}"
        )

    sources = tuple(path.resolve() for path in args.source)
    _runner(args, config).run_many(
        collect_stages(config, extra_sources=sources, render_envs=render_envs)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the E2E workflow command-line interface."""
    parser = build_parser()
    normalized = _normalize_global_args(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(normalized)
    try:
        if args.command == "init":
            if args.config is not None:
                raise ConfigError("--config cannot be combined with init")
            data = build_config(
                run_root=args.run_root,
                sequence_id=args.sequence_id,
                embodiment_contract=args.embodiment_contract,
                task_profile=args.task_profile,
            )
            if args.dry_run:
                print(json.dumps(data, indent=2, sort_keys=True))
                return 0
            config = write_config(data, force=args.force)
            if not args.no_activate:
                set_active(config)
            print(f"Created: {config.path}")
            if not args.no_activate:
                print(f"Active:  {ACTIVE_POINTER} -> {config.path}")
            return 0

        if args.command == "use":
            if args.config is not None:
                raise ConfigError("--config cannot be combined with use")
            config = load_config(args.path)
            if args.dry_run:
                print(f"Would activate: {config.path}")
            else:
                set_active(config)
                print(f"Active: {ACTIVE_POINTER} -> {config.path}")
            return 0

        config = resolve_config(args.config)
        if args.command == "status":
            _status(config, as_json=args.as_json)
        elif args.command == "setup":
            mano = _input_value(args, config, "setup", "mano_dir")
            groot = _input_value(args, config, "setup", "isaac_groot_dir")
            image = _stage_value(
                args,
                config,
                "setup",
                "image_version",
                default=config.runtime["image_version"],
            )
            gpu = int(
                _stage_value(
                    args, config, "setup", "gpu", default=config.runtime["gpu"]
                )
            )
            skip_reconstruction = bool(
                _stage_value(
                    args,
                    config,
                    "setup",
                    "skip_reconstruction",
                    default=False,
                )
            )
            accept_nvidia_model_eula = bool(
                _stage_value(
                    args,
                    config,
                    "setup",
                    "accept_nvidia_model_eula",
                    default=False,
                )
            )
            parameters = {
                "mano_dir": mano,
                "isaac_groot_dir": groot,
                "image_version": image,
                "gpu": gpu,
                "skip_reconstruction": skip_reconstruction,
                "accept_nvidia_model_eula": accept_nvidia_model_eula,
            }
            config = _configure(
                args,
                config,
                "setup",
                parameters,
                inputs={"mano_dir": mano, "isaac_groot_dir": groot},
                runtime={"image_version": image, "gpu": gpu},
            )
            _runner(args, config).run_many(
                setup_stages(
                    config,
                    skip_reconstruction=skip_reconstruction,
                    accept_nvidia_model_eula=accept_nvidia_model_eula,
                )
            )
        elif args.command == "doctor":
            _runner(args, config).run_many(doctor_stages(config))
        elif args.command == "reconstruct":
            video = _input_value(args, config, "reconstruct", "video")
            prompt = config.workflow["object_prompt"]
            hand_tracking = _stage_value(
                args, config, "reconstruct", "hand_tracking", default="hamer"
            )
            run_gsplat_refinement = bool(
                _stage_value(
                    args,
                    config,
                    "reconstruct",
                    "run_gsplat_refinement",
                    default=False,
                )
            )
            dev = bool(_stage_value(args, config, "reconstruct", "dev", default=False))
            if hand_tracking == "dynhamr" and run_gsplat_refinement:
                raise ConfigError(
                    "--run-gsplat-refinement is only supported with "
                    "--hand-tracking hamer"
                )
            parameters = {
                "video": video,
                "object_prompt": prompt,
                "hand_tracking": hand_tracking,
                "run_gsplat_refinement": run_gsplat_refinement,
                "dev": dev,
            }
            config = _configure(
                args,
                config,
                "reconstruct",
                parameters,
                inputs={"video": video},
                workflow={
                    "object_prompt": prompt,
                    "hand_tracking": hand_tracking,
                },
            )
            _runner(args, config).run_many(reconstruct_stages(config))
        elif args.command == "import-reconstruction":
            bundle = Path(
                _stage_value(
                    args,
                    config,
                    "import_reconstruction",
                    "bundle",
                    required=True,
                )
            )
            config = _configure(
                args,
                config,
                "import_reconstruction",
                {"bundle": bundle},
                inputs={"reconstruction_bundle": bundle},
            )
            _runner(args, config).run_many(import_reconstruction_stages(config))
        elif args.command == "inspect":
            _runner(args, config).run_many(inspect_stages(config))
        elif args.command == "retarget":
            sequence = str(config.workflow["sequence_id"])
            embodiment = _stage_value(
                args, config, "retarget", "embodiment", default="vega"
            )
            config = _configure(
                args,
                config,
                "retarget",
                {"sequence_id": sequence, "embodiment": embodiment},
                workflow={"sequence_id": sequence},
            )
            _runner(args, config).run_many(retarget_stages(config, embodiment))
        elif args.command == "simulate":
            embodiment = _stage_value(
                args, config, "simulate", "embodiment", default="vega"
            )
            config = _configure(
                args,
                config,
                "simulate",
                {"embodiment": embodiment},
            )
            _runner(args, config).run_many(simulate_stages(config, embodiment))
        elif args.command == "train-expert":
            embodiment = _stage_value(
                args, config, "train_expert", "embodiment", default="vega"
            )
            max_iterations = int(
                _stage_value(args, config, "train_expert", "max_iterations", default=1)
            )
            num_envs = int(
                _stage_value(args, config, "train_expert", "num_envs", default=1)
            )
            config = _configure(
                args,
                config,
                "train_expert",
                {
                    "embodiment": embodiment,
                    "max_iterations": max_iterations,
                    "num_envs": num_envs,
                },
            )
            _runner(args, config).run_many(
                train_expert_stages(
                    config,
                    embodiment,
                    max_iterations=max_iterations,
                    num_envs=num_envs,
                )
            )
        elif args.command == "collect":
            _run_collect(args, config)
        elif args.command == "finetune":
            instruction = str(config.workflow["instruction"])
            epochs = float(
                _stage_value(args, config, "finetune", "epochs", default=1.0)
            )
            batch = int(
                _stage_value(args, config, "finetune", "global_batch_size", default=32)
            )
            action_horizon = int(
                config.data["contracts"]["embodiment"]["action_horizon"]
            )
            fps = int(config.workflow["fps"])
            base_model = _stage_value(
                args,
                config,
                "finetune",
                "base_model",
                default="nvidia/GR00T-N1.7-3B",
            )
            parameters = {
                "instruction": instruction,
                "epochs": epochs,
                "global_batch_size": batch,
                "action_horizon": action_horizon,
                "fps": fps,
                "base_model": base_model,
            }
            config = _configure(
                args,
                config,
                "finetune",
                parameters,
                workflow={
                    key: value
                    for key, value in parameters.items()
                    if key not in {"instruction", "fps"}
                },
            )
            runner = _runner(args, config)
            if args.dry_run:
                runner.run_many(
                    finetune_stages(
                        config, checkpoint=args.checkpoint, include_open_loop=True
                    )
                )
            else:
                runner.run_many(finetune_stages(config, include_open_loop=False))
                checkpoint = (
                    args.checkpoint.resolve()
                    if args.checkpoint is not None
                    else discover_checkpoint(config.host_path("finetune"))
                )
                runner.run(finetune_stages(config, checkpoint=checkpoint)[-1])
                print(f"Checkpoint: {checkpoint}")
        elif args.command == "evaluate":
            episodes = int(
                _stage_value(args, config, "evaluate", "episodes", default=20)
            )
            default_envs = config.stage("collect").get("render_envs", 1)
            num_envs = int(
                _stage_value(args, config, "evaluate", "num_envs", default=default_envs)
            )
            execution_length = int(
                _stage_value(args, config, "evaluate", "execution_length", default=4)
            )
            motion_source = str(
                _stage_value(args, config, "evaluate", "motion_source", default="run")
            )
            model_seed = int(
                _stage_value(args, config, "evaluate", "model_seed", default=0)
            )
            evaluation_horizon = int(
                _stage_value(
                    args,
                    config,
                    "evaluate",
                    "evaluation_horizon",
                    default=config.workflow.get("frames_per_episode"),
                    required="frames_per_episode" not in config.workflow,
                )
            )
            config = _configure(
                args,
                config,
                "evaluate",
                {
                    "episodes": episodes,
                    "num_envs": num_envs,
                    "execution_length": execution_length,
                    "motion_source": motion_source,
                    "model_seed": model_seed,
                    "evaluation_horizon": evaluation_horizon,
                },
            )
            checkpoint = (
                args.checkpoint.resolve()
                if args.checkpoint is not None
                else (
                    config.host_path("finetune") / "<latest-checkpoint>"
                    if args.dry_run
                    else discover_checkpoint(config.host_path("finetune"))
                )
            )
            _runner(args, config).run_many(
                evaluate_stages(
                    config,
                    checkpoint=checkpoint,
                    episodes=episodes,
                    num_envs=num_envs,
                    motion_source=motion_source,
                    model_seed=model_seed,
                    execution_length=execution_length,
                    evaluation_horizon=evaluation_horizon,
                )
            )
        else:  # pragma: no cover - argparse guarantees a known command
            parser.error(f"unknown command: {args.command}")
        return 0
    except (ConfigError, StageError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
