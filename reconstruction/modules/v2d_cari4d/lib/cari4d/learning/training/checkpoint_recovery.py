from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path

from learning.training.resume_state import rank_rng_state_path


CHECKPOINT_RECOVERY_SCHEMA = "cari4d.training_checkpoint_recovery.v1"
CHECKPOINT_STEP_RE = re.compile(r"^step(\d+)\.pth$")
REQUIRED_CHECKPOINT_KEYS = ("model", "optimizer", "scheduler", "epoch", "step", "epoch_step", "amp_scaler", "rng_state", "cfg")
REQUIRED_RNG_KEYS = ("python_random", "numpy_random", "torch_cpu")


def checkpoint_recovery_certificate_path(checkpoint_path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}.recovery.json")


def _path_identity(path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file() or stat.st_size < 1:
        raise ValueError(f"checkpoint recovery artifact is empty or not a file: {path}")
    return {"path": str(path), "size": int(stat.st_size), "mtimeNs": int(stat.st_mtime_ns)}


def _canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _checkpoint_step(checkpoint_path) -> int:
    match = CHECKPOINT_STEP_RE.match(Path(checkpoint_path).name)
    if match is None:
        raise ValueError(f"checkpoint filename does not encode a training step: {checkpoint_path}")
    return int(match.group(1))


def validate_checkpoint_payload(checkpoint_payload, checkpoint_path) -> int:
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("training checkpoint payload must be a mapping")
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint_payload]
    if missing:
        raise ValueError(f"training checkpoint is missing required state: {missing}")
    step = checkpoint_payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"training checkpoint step must be a nonnegative integer, got {step}")
    filename_step = _checkpoint_step(checkpoint_path)
    if step != filename_step:
        raise ValueError(f"training checkpoint step {step} does not match filename step {filename_step}")
    for key in ("model", "optimizer", "scheduler"):
        if not isinstance(checkpoint_payload[key], Mapping) or not checkpoint_payload[key]:
            raise ValueError(f"training checkpoint {key} state must be a nonempty mapping")
    if not checkpoint_payload["optimizer"].get("param_groups"):
        raise ValueError("training checkpoint optimizer state has no parameter groups")
    for key in ("epoch", "epoch_step"):
        value = checkpoint_payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"training checkpoint {key} must be a nonnegative integer, got {value}")
    rng_state = checkpoint_payload["rng_state"]
    if not isinstance(rng_state, Mapping) or any(key not in rng_state for key in REQUIRED_RNG_KEYS):
        raise ValueError("training checkpoint random-number-generator state is incomplete")
    return step


def _sidecar_identities(checkpoint_path, world_size: int) -> list[dict]:
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"checkpoint recovery world size must be positive, got {world_size}")
    return [_path_identity(rank_rng_state_path(checkpoint_path, rank)) for rank in range(world_size)]


def publish_checkpoint_recovery_certificate(checkpoint_path, checkpoint_payload, world_size: int) -> dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    step = validate_checkpoint_payload(checkpoint_payload, checkpoint_path)
    core = {
        "schema": CHECKPOINT_RECOVERY_SCHEMA,
        "checkpoint": _path_identity(checkpoint_path),
        "step": step,
        "worldSize": int(world_size),
        "requiredState": list(REQUIRED_CHECKPOINT_KEYS),
        "rankRngSidecars": _sidecar_identities(checkpoint_path, world_size),
        "checkpointReason": str(checkpoint_payload.get("checkpoint_reason", "")),
    }
    payload = {**core, "certificateId": _canonical_hash(core)}
    certificate_path = checkpoint_recovery_certificate_path(checkpoint_path)
    tmp_path = certificate_path.with_name(f".{certificate_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp_path, certificate_path)
    return {**payload, "certificatePath": str(certificate_path)}


def validate_checkpoint_recovery_certificate(checkpoint_path) -> dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    certificate_path = checkpoint_recovery_certificate_path(checkpoint_path)
    payload = json.loads(certificate_path.read_text())
    certificate_id = payload.pop("certificateId", None)
    if payload.get("schema") != CHECKPOINT_RECOVERY_SCHEMA:
        raise ValueError(f"unsupported checkpoint recovery certificate schema: {payload.get('schema')}")
    if certificate_id != _canonical_hash(payload):
        raise ValueError(f"checkpoint recovery certificate hash mismatch: {certificate_path}")
    if payload.get("checkpoint") != _path_identity(checkpoint_path):
        raise ValueError(f"checkpoint recovery certificate does not match checkpoint identity: {checkpoint_path}")
    if payload.get("step") != _checkpoint_step(checkpoint_path):
        raise ValueError(f"checkpoint recovery certificate step does not match checkpoint filename: {checkpoint_path}")
    world_size = payload.get("worldSize")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"checkpoint recovery certificate has invalid world size: {world_size}")
    if payload.get("requiredState") != list(REQUIRED_CHECKPOINT_KEYS):
        raise ValueError("checkpoint recovery certificate required-state contract is invalid")
    if payload.get("rankRngSidecars") != _sidecar_identities(checkpoint_path, world_size):
        raise ValueError(f"checkpoint recovery certificate rank sidecars do not match current files: {checkpoint_path}")
    return {**payload, "certificateId": certificate_id, "certificatePath": str(certificate_path)}


def certify_checkpoint_file(checkpoint_path, world_size: int) -> dict:
    import torch

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(checkpoint_payload, checkpoint_path)
    for rank in range(int(world_size)):
        rng_payload = torch.load(rank_rng_state_path(checkpoint_path, rank), map_location="cpu", weights_only=False)
        if not isinstance(rng_payload, Mapping) or any(key not in rng_payload for key in REQUIRED_RNG_KEYS):
            raise ValueError(f"checkpoint rank {rank} random-number-generator sidecar is incomplete")
    return publish_checkpoint_recovery_certificate(checkpoint_path, checkpoint_payload, world_size)
