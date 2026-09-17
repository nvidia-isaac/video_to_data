from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SPEED_BENCHMARK_SCHEMA = "cari4d.training_speed_benchmark.v1"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def benchmark_total_steps(cfg: Any) -> int:
    warmup_steps = int(_cfg_get(cfg, "benchmark_warmup_steps", 0) or 0)
    measured_steps = int(_cfg_get(cfg, "benchmark_steps", 0) or 0)
    if warmup_steps < 0 or measured_steps < 0:
        raise ValueError(f"benchmark step counts must be nonnegative, got warmup={warmup_steps}, measured={measured_steps}")
    return 0 if measured_steps == 0 else warmup_steps + measured_steps


def benchmark_complete(cfg: Any, start_step: int, current_step: int) -> bool:
    total_steps = benchmark_total_steps(cfg)
    return total_steps > 0 and int(current_step) - int(start_step) >= total_steps


def update_benchmark_sample_identity(hasher: Any, metadata_rows: Any) -> None:
    encoded = json.dumps(metadata_rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_speed_benchmark_rank_result(exp_dir: str, rank: int, payload: Mapping[str, Any]) -> Path:
    path = Path(exp_dir) / f"benchmark-rank{int(rank):02d}.json"
    _atomic_write_json(path, payload)
    return path


def _aggregate_timing_scope(rows: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    phase_names = sorted({phase for row in rows for phase in row.get(key, {})})
    output = {}
    for phase in phase_names:
        phase_rows = [row[key][phase] for row in rows if phase in row.get(key, {})]
        counts = [int(item["count"]) for item in phase_rows]
        output[phase] = {"rank_count": len(phase_rows), "sample_count_min": min(counts), "sample_count_max": max(counts)}
        for statistic in ("mean_seconds", "p50_seconds", "p95_seconds", "max_seconds"):
            output[phase][f"critical_rank_{statistic}"] = max(float(item[statistic]) for item in phase_rows)
    return output


def aggregate_speed_benchmark_rank_results(exp_dir: str, world_size: int) -> dict[str, Any]:
    root = Path(exp_dir)
    rows = [json.loads((root / f"benchmark-rank{rank:02d}.json").read_text(encoding="utf-8")) for rank in range(int(world_size))]
    required_equal = ("worker_count", "world_size", "batch_size_per_rank", "clip_length", "warmup_steps", "measured_steps")
    for key in required_equal:
        values = {int(row[key]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"Benchmark rank results disagree on {key}: {sorted(values)}")
    if int(rows[0]["world_size"]) != int(world_size):
        raise ValueError(f"Benchmark result world size {rows[0]['world_size']} != expected {world_size}")
    critical_elapsed = max(float(row["measured_elapsed_seconds"]) for row in rows)
    measured_steps = int(rows[0]["measured_steps"])
    global_batch_size = int(rows[0]["batch_size_per_rank"]) * int(world_size)
    rank_identities = {str(rank): str(row["sample_identity_sha256"]) for rank, row in enumerate(rows)}
    combined_identity = hashlib.sha256(json.dumps(rank_identities, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    result = {
        "schema": SPEED_BENCHMARK_SCHEMA,
        "workerCount": int(rows[0]["worker_count"]),
        "worldSize": int(world_size),
        "batchSizePerRank": int(rows[0]["batch_size_per_rank"]),
        "globalBatchSize": global_batch_size,
        "clipLength": int(rows[0]["clip_length"]),
        "warmupSteps": int(rows[0]["warmup_steps"]),
        "measuredSteps": measured_steps,
        "criticalElapsedSeconds": critical_elapsed,
        "throughputClipsPerSecond": global_batch_size * measured_steps / critical_elapsed,
        "throughputFramesPerSecond": global_batch_size * measured_steps * int(rows[0]["clip_length"]) / critical_elapsed,
        "sampleIdentitySha256": combined_identity,
        "rankSampleIdentitySha256": rank_identities,
        "phases": _aggregate_timing_scope(rows, "phases"),
        "preprocess": _aggregate_timing_scope(rows, "preprocess"),
    }
    _atomic_write_json(root / "benchmark-result.json", result)
    return result
