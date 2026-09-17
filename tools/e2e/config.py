"""Configuration and active-run state for the local E2E launcher."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONFIG_FILENAME = "e2e_config.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTIC_ROOT = REPO_ROOT / "robotic_grounding"
if str(ROBOTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTIC_ROOT))

from groot_finetune.contracts import (  # noqa: E402
    load_embodiment_contract,
)
from groot_finetune.task_profile import load_task_profile  # noqa: E402

ACTIVE_POINTER = REPO_ROOT / ".e2e" / "current"

RECONSTRUCTION_BUNDLE_FILES = (
    Path("result.npz"),
    Path("mesh.obj"),
    Path("manifest.json"),
    Path("threejs_scene/index.html"),
)


class ConfigError(ValueError):
    """Raised when an E2E configuration or active-run pointer is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConfigError(f"{label} is not a file: {path}")


def _require_nonempty_file(path: Path, label: str) -> None:
    _require_file(path, label)
    if path.stat().st_size == 0:
        raise ConfigError(f"{label} is empty: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ConfigError(f"{label} is not a directory: {path}")


def validate_input_path(name: str, path: str | os.PathLike[str]) -> Path:
    """Resolve and validate one stage-owned external input."""
    resolved = _absolute(path)
    if name == "video":
        _require_file(resolved, "input video")
    elif name == "mano_dir":
        _require_dir(resolved, "MANO directory")
        _require_file(resolved / "models/MANO_LEFT.pkl", "left MANO model")
        _require_file(resolved / "models/MANO_RIGHT.pkl", "right MANO model")
    elif name == "isaac_groot_dir":
        _require_dir(resolved, "Isaac-GR00T directory")
    elif name == "rl_checkpoint":
        _require_file(resolved, "RL checkpoint")
    elif name == "reconstruction_bundle":
        _require_dir(resolved, "reconstruction bundle")
        for relative in RECONSTRUCTION_BUNDLE_FILES:
            _require_nonempty_file(
                resolved / relative,
                f"reconstruction bundle file {relative.as_posix()}",
            )
    else:
        raise ConfigError(f"unknown external input: {name}")
    return resolved


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _default_container_name(image_version: str, gpu: int) -> str:
    return f"robotic-grounding-{image_version}-gpu{gpu}"


def _base_paths(run_root: Path) -> dict[str, dict[str, str]]:
    container_run = Path("/workspace/e2e")
    return {
        "host": {
            "reconstruction": str(run_root / "reconstruction"),
            "bundle": str(run_root / "reconstruction" / "result_slam_gravity_aligned"),
            "ego_raw": str(run_root / "ego_recon_raw"),
            "loaded": str(run_root / "intermediate/ego_recon/loaded"),
            "human_motion_data": str(run_root / "human_motion_data"),
            "processed": str(run_root / "human_motion_data/ego_recon/processed"),
            "source": str(run_root / "source"),
            "selected": str(run_root / "selected"),
            "recording": str(run_root / "recording"),
            "dataset": str(run_root / "gr00t_dataset"),
            "finetune": str(run_root / "finetune"),
            "open_loop": str(run_root / "open_loop"),
            "closed_loop": str(run_root / "closed_loop"),
            "embodiment_contract": str(run_root / "contracts/embodiment.json"),
            "task_profile": str(run_root / "contracts/task_profile.json"),
        },
        "container": {
            "repo_root": "/workspace/video_to_data",
            "run_root": str(container_run),
            "mano_dir": "/workspace/mano",
            "bundle": str(container_run / "reconstruction/result_slam_gravity_aligned"),
            "loaded": str(container_run / "intermediate/ego_recon/loaded"),
            "human_motion_data": str(container_run / "human_motion_data"),
            "processed": str(container_run / "human_motion_data/ego_recon/processed"),
            "source": str(container_run / "source"),
            "selected": str(container_run / "selected"),
            "recording": str(container_run / "recording"),
            "dataset": str(container_run / "gr00t_dataset"),
            "closed_loop": str(container_run / "closed_loop"),
            "embodiment_contract": str(container_run / "contracts/embodiment.json"),
            "task_profile": str(container_run / "contracts/task_profile.json"),
        },
    }


def build_config(
    *,
    run_root: str | os.PathLike[str],
    sequence_id: str,
    embodiment_contract: str | os.PathLike[str],
    task_profile: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a durable run configuration from explicit post-training contracts."""
    host_run_root = _absolute(run_root)
    _validate_sequence_id(sequence_id)
    contract = load_embodiment_contract(embodiment_contract)
    profile = load_task_profile(task_profile)
    workflow = {
        "sequence_id": sequence_id,
        "embodiment_contract": contract.contract_id,
        "embodiment_contract_sha256": contract.sha256,
        "task_profile": profile.task_id,
        "task_profile_sha256": profile.sha256,
        "object_prompt": profile.object_prompt,
        "instruction": profile.instruction,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "camera_count": len(contract.cameras),
        "fps": contract.fps,
        "action_horizon": contract.action_horizon,
    }
    paths = _base_paths(host_run_root)
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "run_root": str(host_run_root),
        "inputs": {},
        "runtime": {
            "gpu": 0,
            "image_version": "latest",
            "container_name": _default_container_name("latest", 0),
        },
        "workflow": workflow,
        "contracts": {
            "embodiment": contract.as_dict(),
            "task": profile.as_dict(),
        },
        "stage_parameters": {},
        "paths": paths,
    }
    validate_config_data(data)
    return data


def _validate_sequence_id(value: str) -> None:
    if not value.strip() or "/" in value or "\\" in value:
        raise ConfigError("sequence ID must be a non-empty path component")


def validate_config_data(data: Mapping[str, Any]) -> None:
    """Validate the durable schema without requiring unconfigured stage inputs."""
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"unsupported schema_version={data.get('schema_version')!r}; expected {SCHEMA_VERSION}")
    for key in (
        "repo_root",
        "run_root",
        "inputs",
        "runtime",
        "workflow",
        "contracts",
        "paths",
    ):
        if key not in data:
            raise ConfigError(f"configuration is missing {key!r}")
    repo_root = Path(str(data["repo_root"]))
    run_root = Path(str(data["run_root"]))
    if not repo_root.is_absolute() or not run_root.is_absolute():
        raise ConfigError("repo_root and run_root must be absolute")
    if repo_root.resolve() != REPO_ROOT:
        raise ConfigError(f"configuration belongs to a different checkout: {repo_root} (expected {REPO_ROOT})")
    inputs = data["inputs"]
    if not isinstance(inputs, Mapping):
        raise ConfigError("inputs must be an object")
    for key, value in inputs.items():
        if key not in {
            "video",
            "mano_dir",
            "isaac_groot_dir",
            "rl_checkpoint",
            "reconstruction_bundle",
        }:
            raise ConfigError(f"unknown configured input: {key}")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ConfigError(f"inputs.{key} must be an absolute path")
    stage_parameters = data.get("stage_parameters", {})
    if not isinstance(stage_parameters, Mapping):
        raise ConfigError("stage_parameters must be an object")
    runtime = data["runtime"]
    if not isinstance(runtime, Mapping):
        raise ConfigError("runtime must be an object")
    if not str(runtime.get("image_version", "")).strip():
        raise ConfigError("runtime image version cannot be empty")
    if int(runtime.get("gpu", -1)) < 0:
        raise ConfigError("GPU index must be non-negative")
    workflow = data["workflow"]
    if not isinstance(workflow, Mapping):
        raise ConfigError("workflow must be an object")
    for key, label in (
        ("object_prompt", "object prompt"),
        ("instruction", "language instruction"),
        ("base_model", "base model"),
    ):
        if key in workflow and not str(workflow[key]).strip():
            raise ConfigError(f"{label} cannot be empty")
    if "sequence_id" in workflow:
        _validate_sequence_id(str(workflow["sequence_id"]))
    contracts = data["contracts"]
    if not isinstance(contracts, Mapping):
        raise ConfigError("contracts must be an object")
    try:
        contract = load_embodiment_contract(contracts["embodiment"])
        profile = load_task_profile(contracts["task"])
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"invalid contracts: {exc}") from exc
    expected_workflow = {
        "embodiment_contract": contract.contract_id,
        "embodiment_contract_sha256": contract.sha256,
        "task_profile": profile.task_id,
        "task_profile_sha256": profile.sha256,
        "object_prompt": profile.object_prompt,
        "instruction": profile.instruction,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "camera_count": len(contract.cameras),
        "fps": contract.fps,
        "action_horizon": contract.action_horizon,
    }
    mismatches = {
        key: (workflow.get(key), expected)
        for key, expected in expected_workflow.items()
        if workflow.get(key) != expected
    }
    if mismatches:
        raise ConfigError(f"workflow does not match its contracts: {mismatches}")
    if "hand_tracking" in workflow and workflow["hand_tracking"] not in {
        "hamer",
        "dynhamr",
    }:
        raise ConfigError("hand tracking must be 'hamer' or 'dynhamr'")
    for key in (
        "target_successes",
        "pilot_attempts",
        "collection_max_steps",
        "evaluation_horizon",
        "frames_per_episode",
        "render_envs",
        "action_horizon",
        "global_batch_size",
        "fps",
        "state_dim",
        "action_dim",
        "camera_count",
    ):
        if key in workflow and float(workflow[key]) <= 0:
            raise ConfigError(f"workflow.{key} must be positive")
    for key in ("collection_safety_factor", "epochs"):
        if key in workflow and float(workflow[key]) <= 0:
            raise ConfigError(f"workflow.{key} must be positive")
    for key in (
        "reset_arm_noise_rad",
        "reset_finger_noise_rad",
        "reset_object_xy_noise_m",
        "reset_object_yaw_noise_rad",
    ):
        if key in workflow and float(workflow[key]) < 0:
            raise ConfigError(f"workflow.{key} must be non-negative")
    if "measured_success_rate" in workflow and not (0.0 < float(workflow["measured_success_rate"]) <= 1.0):
        raise ConfigError("measured success rate must be in (0, 1]")
    paths = data["paths"]
    if not isinstance(paths, Mapping):
        raise ConfigError("paths must be an object")
    for namespace in ("host", "container"):
        mapped = paths.get(namespace)
        if not isinstance(mapped, Mapping) or not mapped:
            raise ConfigError(f"paths.{namespace} must be a non-empty object")
        for key, value in mapped.items():
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ConfigError(f"paths.{namespace}.{key} must be an absolute path")


@dataclass(frozen=True)
class E2EConfig:
    """Loaded E2E configuration with typed path and stage accessors."""

    path: Path
    data: Mapping[str, Any]

    @property
    def run_root(self) -> Path:
        return Path(str(self.data["run_root"]))

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self.data["inputs"]  # type: ignore[return-value]

    @property
    def workflow(self) -> Mapping[str, Any]:
        return self.data["workflow"]  # type: ignore[return-value]

    @property
    def runtime(self) -> Mapping[str, Any]:
        return self.data["runtime"]  # type: ignore[return-value]

    def stage(self, name: str) -> Mapping[str, Any]:
        stages = self.data.get("stage_parameters", {})
        value = stages.get(name, {}) if isinstance(stages, Mapping) else {}
        return value if isinstance(value, Mapping) else {}

    def host_path(self, name: str) -> Path:
        return Path(str(self.data["paths"]["host"][name]))  # type: ignore[index]

    def container_path(self, name: str) -> Path:
        try:
            value = self.data["paths"]["container"][name]  # type: ignore[index]
        except KeyError as exc:
            raise ConfigError(f"container path {name!r} is not configured yet") from exc
        return Path(str(value))

    def has_input(self, name: str) -> bool:
        return name in self.inputs

    def input_path(self, name: str) -> Path:
        try:
            value = self.inputs[name]
        except KeyError as exc:
            raise ConfigError(f"{name.replace('_', ' ')} is not configured; provide it at its owning stage") from exc
        return Path(str(value))

    def validate_inputs(self, names: tuple[str, ...] | None = None) -> None:
        selected = names if names is not None else tuple(self.inputs)
        for name in selected:
            validate_input_path(name, self.input_path(name))

    def processed_path(self, *, container: bool = False) -> Path:
        """Return the configured root of the Hive-partitioned motion dataset.

        Parquet writers append ``sequence_id=<id>/robot_name=<robot>`` themselves,
        so the workflow must not insert another sequence directory here.
        """
        return self.container_path("processed") if container else self.host_path("processed")


def write_config(data: Mapping[str, Any], *, force: bool = False) -> E2EConfig:
    validate_config_data(data)
    path = Path(str(data["run_root"])) / CONFIG_FILENAME
    if path.exists() and not force:
        raise ConfigError(f"configuration already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = data["paths"]["host"]  # type: ignore[index]
    _atomic_json(Path(str(paths["embodiment_contract"])), data["contracts"]["embodiment"])  # type: ignore[index]
    _atomic_json(Path(str(paths["task_profile"])), data["contracts"]["task"])  # type: ignore[index]
    _atomic_json(path, data)
    return E2EConfig(path.resolve(), dict(data))


def update_stage_config(
    config: E2EConfig,
    stage: str,
    parameters: Mapping[str, Any],
    *,
    inputs: Mapping[str, str | os.PathLike[str]] | None = None,
    runtime: Mapping[str, Any] | None = None,
    workflow: Mapping[str, Any] | None = None,
    persist: bool = True,
) -> E2EConfig:
    """Persist the resolved parameters owned by one stage."""
    data = copy.deepcopy(dict(config.data))
    data.setdefault("stage_parameters", {})[stage] = {
        key: str(value) if isinstance(value, Path) else value for key, value in parameters.items()
    }
    if inputs:
        for name, value in inputs.items():
            data["inputs"][name] = str(validate_input_path(name, value))
    if runtime:
        data["runtime"].update(runtime)
        image = str(data["runtime"]["image_version"])
        gpu = int(data["runtime"]["gpu"])
        if gpu < 0:
            raise ConfigError("GPU index must be non-negative")
        data["runtime"]["container_name"] = _default_container_name(image, gpu)
    if workflow:
        for key, value in workflow.items():
            if value is None:
                data["workflow"].pop(key, None)
            else:
                data["workflow"][key] = value
    if "sequence_id" in data["workflow"]:
        _validate_sequence_id(str(data["workflow"]["sequence_id"]))
    checkpoint = data["inputs"].get("rl_checkpoint")
    if checkpoint:
        data["paths"]["container"]["rl_checkpoint"] = str(Path("/workspace/checkpoints") / Path(checkpoint).name)
    validate_config_data(data)
    comparable = copy.deepcopy(data)
    comparable["updated_at"] = config.data.get("updated_at")
    if comparable == config.data:
        return config
    data["updated_at"] = _utc_now()
    if persist:
        _atomic_json(config.path, data)
    return E2EConfig(config.path, data)


def load_config(path: str | os.PathLike[str]) -> E2EConfig:
    candidate = _absolute(path)
    if candidate.is_dir():
        candidate = candidate / CONFIG_FILENAME
    if not candidate.is_file():
        raise ConfigError(f"configuration does not exist: {candidate}")
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read configuration {candidate}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"configuration must contain a JSON object: {candidate}")
    data.setdefault("stage_parameters", {})
    data.setdefault("updated_at", data.get("created_at", _utc_now()))
    validate_config_data(data)
    expected = Path(str(data["run_root"])) / CONFIG_FILENAME
    if candidate != expected.resolve():
        raise ConfigError(f"configuration must be stored at its run root: {expected} (got {candidate})")
    for label, path_name, embedded_name, loader in (
        (
            "embodiment contract",
            "embodiment_contract",
            "embodiment",
            load_embodiment_contract,
        ),
        ("task profile", "task_profile", "task", load_task_profile),
    ):
        snapshot = Path(str(data["paths"]["host"][path_name]))
        if not snapshot.is_file():
            raise ConfigError(f"missing {label} snapshot: {snapshot}")
        try:
            actual = loader(snapshot)
            embedded = loader(data["contracts"][embedded_name])
        except ValueError as exc:
            raise ConfigError(f"invalid {label} snapshot: {exc}") from exc
        if actual.sha256 != embedded.sha256:
            raise ConfigError(f"{label} snapshot differs from the active run configuration: {snapshot}")
    return E2EConfig(candidate, data)


def set_active(config: E2EConfig) -> None:
    ACTIVE_POINTER.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".current.", dir=ACTIVE_POINTER.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(config.path) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ACTIVE_POINTER)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def active_config() -> E2EConfig:
    if not ACTIVE_POINTER.is_file():
        raise ConfigError("no active E2E run; run './run_e2e.sh init --run-root ...' or pass --config")
    value = ACTIVE_POINTER.read_text(encoding="utf-8").strip()
    if not value:
        raise ConfigError(f"active-run pointer is empty: {ACTIVE_POINTER}")
    return load_config(value)


def resolve_config(override: str | os.PathLike[str] | None) -> E2EConfig:
    return load_config(override) if override is not None else active_config()
