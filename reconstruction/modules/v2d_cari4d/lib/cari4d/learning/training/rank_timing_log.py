from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO


class TimingWindow:
    def __init__(self):
        self.values = defaultdict(list)

    def add(self, phase: str, seconds: float) -> None:
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError(f"Timing values must be nonnegative, got {phase}={seconds}")
        self.values[str(phase)].append(seconds)

    def summary(self, reset: bool = False) -> dict[str, dict[str, float | int]]:
        output = {}
        for phase, values in sorted(self.values.items()):
            ordered = sorted(values)
            p50_index = int(round(0.50 * (len(ordered) - 1)))
            p95_index = int(round(0.95 * (len(ordered) - 1)))
            output[phase] = {"count": len(ordered), "mean_seconds": sum(ordered) / len(ordered), "p50_seconds": ordered[p50_index], "p95_seconds": ordered[p95_index], "max_seconds": ordered[-1]}
        if reset:
            self.values.clear()
        return output


def write_rank_timing_record(handle: TextIO | None, record_type: str, payload: dict[str, Any]) -> None:
    line = f"[{str(record_type)}] {json.dumps(payload, sort_keys=True)}"
    if handle is None:
        print(line, flush=True)
    else:
        handle.write(line + "\n")


def open_rank_timing_log(directory: str | None, rank: int) -> TextIO | None:
    if not directory:
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    return (root / f"rank{int(rank):02d}.log").open("a", encoding="utf-8", buffering=1)
