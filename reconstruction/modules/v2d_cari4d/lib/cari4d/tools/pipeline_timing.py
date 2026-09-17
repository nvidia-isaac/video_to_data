from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any


MHR_PIPELINE_TIMING_PREFIX = "MHR_PIPELINE_TIMING "
MHR_PIPELINE_TIMING_SCHEMA = "cari4d.mhr_pipeline_timing.v1"


class PipelineTimer:
    """Low-overhead wall-clock aggregation for coarse inference phases."""

    def __init__(self, component: str, *, clock: Callable[[], float] = time.perf_counter):
        if not component:
            raise ValueError("Pipeline timing component must be nonempty")
        self.component = str(component)
        self._clock = clock
        self._started = self._clock()
        self._phase_seconds: dict[str, float] = defaultdict(float)
        self._phase_calls: dict[str, int] = defaultdict(int)
        self._metadata: dict[str, Any] = {}
        self._emitted = False

    def start(self) -> float:
        return self._clock()

    def elapsed(self, started: float) -> float:
        elapsed = self._clock() - float(started)
        if elapsed < 0:
            raise ValueError(f"Pipeline timing clock moved backwards: {elapsed}")
        return elapsed

    def record_elapsed(self, phase: str, elapsed: float) -> float:
        if not phase:
            raise ValueError("Pipeline timing phase must be nonempty")
        elapsed = float(elapsed)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"Pipeline timing elapsed seconds must be finite and nonnegative for {phase!r}, got {elapsed}")
        self._phase_seconds[str(phase)] += elapsed
        self._phase_calls[str(phase)] += 1
        return elapsed

    def record(self, phase: str, started: float) -> float:
        return self.record_elapsed(phase, self.elapsed(started))

    def update_metadata(self, values: Mapping[str, Any]) -> None:
        self._metadata.update(values)

    def payload(self, *, status: str) -> dict[str, Any]:
        total_seconds = self._clock() - self._started
        phases = {name: {"seconds": round(self._phase_seconds[name], 6), "calls": self._phase_calls[name]} for name in sorted(self._phase_seconds)}
        return {"schema": MHR_PIPELINE_TIMING_SCHEMA, "component": self.component, "status": str(status), "total_seconds": round(total_seconds, 6), "phases": phases, "metadata": dict(sorted(self._metadata.items()))}

    def emit(self, *, status: str) -> dict[str, Any]:
        if self._emitted:
            raise RuntimeError(f"Pipeline timing was already emitted for {self.component}")
        payload = self.payload(status=status)
        print(MHR_PIPELINE_TIMING_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
        self._emitted = True
        return payload

    def __enter__(self) -> "PipelineTimer":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: object | None) -> bool:
        self.emit(status="complete" if exc_type is None else "failed")
        return False
