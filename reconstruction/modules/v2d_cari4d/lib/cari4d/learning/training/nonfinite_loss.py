from __future__ import annotations

import json
import math
import os
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from learning.datasets.mhr_sample_metadata import MHR_SAMPLE_METADATA_KEY


@dataclass(frozen=True)
class LossGuardContext:
    exp_dir: str
    global_step: int
    epoch: int
    epoch_batch_index: int
    rank: int
    world_size: int
    experiment_name: str | None
    seed: int | None
    resume_checkpoint: str | None


def _loss_entries(loss: Any, loss_dict: Mapping[str, Any]) -> list[tuple[str, Any]]:
    entries = [(name, loss_dict[name]) for name in sorted(loss_dict) if name.startswith("loss") and name != "loss"]
    entries.append(("loss", loss))
    return [(name, value) for name, value in entries if torch.is_tensor(value) or isinstance(value, Number)]


def _reference_device(entries: list[tuple[str, Any]]) -> torch.device:
    for _name, value in entries:
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _nonfinite_flag(entries: list[tuple[str, Any]]) -> torch.Tensor:
    device = _reference_device(entries)
    flag = torch.zeros((), dtype=torch.int32, device=device)
    for _name, value in entries:
        if torch.is_tensor(value):
            bad = torch.logical_not(torch.isfinite(value.detach()).all()).to(device=device, dtype=torch.int32)
            flag = torch.maximum(flag, bad)
        elif not math.isfinite(float(value)):
            flag.fill_(1)
    return flag


def _nonfinite_value(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    return repr(value)


def _flat_index_to_coordinates(index: int, shape: tuple[int, ...]) -> list[int]:
    coordinates = []
    for size in reversed(shape):
        coordinates.append(index % size)
        index //= size
    return list(reversed(coordinates))


def _local_nonfinite_losses(entries: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for name, value in entries:
        if torch.is_tensor(value):
            detached = value.detach()
            finite = torch.isfinite(detached)
            if bool(finite.all().item()):
                continue
            flat_index = int(torch.nonzero(torch.logical_not(finite.reshape(-1)), as_tuple=False)[0, 0].item())
            shape = tuple(int(size) for size in detached.shape)
            failures.append({"name": name, "value": _nonfinite_value(detached.reshape(-1)[flat_index].item()), "shape": list(shape), "index": _flat_index_to_coordinates(flat_index, shape)})
        elif not math.isfinite(float(value)):
            failures.append({"name": name, "value": _nonfinite_value(float(value)), "shape": [], "index": []})
    return failures


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return _nonfinite_value(value) if not math.isfinite(value) else value
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _sequence_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _batch_metadata(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = batch.get(MHR_SAMPLE_METADATA_KEY)
    if metadata is not None:
        if not isinstance(metadata, (list, tuple)) or not all(isinstance(item, Mapping) for item in metadata):
            raise TypeError(f"{MHR_SAMPLE_METADATA_KEY} must contain an ordered list of mappings")
        return [_json_safe(dict(item)) for item in metadata]
    return [{"sequence_name": name} for name in _sequence_names(batch.get("seq_name"))]


def _distributed_active(context: LossGuardContext) -> bool:
    active = dist.is_available() and dist.is_initialized()
    if active:
        if dist.get_rank() != context.rank or dist.get_world_size() != context.world_size:
            raise RuntimeError(f"Loss guard distributed context mismatch: context rank/world={context.rank}/{context.world_size}, process group={dist.get_rank()}/{dist.get_world_size()}")
    elif context.world_size != 1:
        raise RuntimeError(f"Loss guard expected world_size={context.world_size}, but torch.distributed is not initialized")
    return active


def _gather_rank_reports(local_report: dict[str, Any], context: LossGuardContext, distributed: bool) -> list[dict[str, Any]]:
    if not distributed:
        return [local_report]
    reports: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(reports, local_report)
    if any(report is None for report in reports):
        raise RuntimeError("Loss guard failed to gather diagnostics from every rank")
    return [report for report in reports if report is not None]


def _failure_record_path(context: LossGuardContext) -> Path:
    directory = Path(context.exp_dir) / "nonfinite-loss-events"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step{context.global_step:08d}-epoch{context.epoch:06d}-batch{context.epoch_batch_index:06d}-observer-rank{context.rank:05d}-event{time.time_ns()}.json"
    return directory / name


def _write_failure_record(record: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _failure_message(failures: list[dict[str, Any]], context: LossGuardContext) -> str:
    primary = failures[0]
    primary_loss = primary["losses"][0]
    sequence_names = sorted({sample.get("sequence_name") for failure in failures for sample in failure["samples"] if sample.get("sequence_name")})
    sequences = ",".join(sequence_names) if sequence_names else "unavailable"
    failure_text = ",".join(f"rank{failure['rank']}:{loss['name']}={loss['value']}" for failure in failures for loss in failure["losses"])
    return f"Nonfinite loss detected before backward: {primary_loss['name']}={primary_loss['value']} failing_rank={primary['rank']} global_step={context.global_step} epoch={context.epoch} epoch_batch_index={context.epoch_batch_index} sequences={sequences} failures={failure_text}"


def guard_finite_losses(loss: Any, loss_dict: Mapping[str, Any], batch: Mapping[str, Any], context: LossGuardContext) -> None:
    entries = _loss_entries(loss, loss_dict)
    local_flag = _nonfinite_flag(entries)
    distributed = _distributed_active(context)
    global_flag = local_flag.clone()
    if distributed:
        dist.all_reduce(global_flag, op=dist.ReduceOp.MAX)
    if not bool(global_flag.item()):
        return
    local_report = {"rank": context.rank, "host": socket.gethostname(), "pid": os.getpid(), "device": str(_reference_device(entries)), "losses": _local_nonfinite_losses(entries), "samples": _batch_metadata(batch)}
    rank_reports = _gather_rank_reports(local_report, context, distributed)
    failures = sorted((report for report in rank_reports if report["losses"]), key=lambda report: int(report["rank"]))
    if not failures:
        raise RuntimeError("Loss guard observed a global nonfinite flag without rank diagnostics")
    record = {
        "schema": "cari4d.nonfinite_loss.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "observer_rank": context.rank,
        "observer_host": socket.gethostname(),
        "observer_pid": os.getpid(),
        "context": asdict(context),
        "failures": failures,
        "rank_reports": rank_reports,
    }
    path = _failure_record_path(context)
    _write_failure_record(record, path)
    message = _failure_message(failures, context)
    print(f"NONFINITE_LOSS {message} record={path}", flush=True)
    raise RuntimeError(f"{message} record={path}")
