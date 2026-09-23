# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock stage timing for the reconstruction pipelines.

The ego pipelines announce their stages with ``[run ]``/``[skip]`` markers but do
not measure them. This module turns those markers into a timeline.

Two levels are recorded:

- **Stage**: bounded by consecutive ``_step`` markers, so a stage's duration runs
  from its own marker to the next one (or to the end of the run). It therefore
  includes the host-side glue that follows the stage's real work -- overlay
  rendering, JSON bookkeeping, interpolation -- which is what a timestamped log
  would attribute to it as well.
- **Container**: the exact wall time of each ``docker run`` issued while a stage
  is open. Comparing ``container_seconds`` against a stage's total separates GPU
  work inside the image from host-side work around it.

Durations use :func:`time.perf_counter` -- a monotonic clock that cannot be
dragged backwards by NTP adjustments mid-run. Epoch timestamps recorded
alongside come from :func:`time.time` and exist only to anchor the timeline to
a human-readable date. Host CPU time (:func:`time.process_time`) is recorded for
the run as a whole: with every heavy stage executing in a child container, that
number should stay small, and a large value means host-side work is a real cost
rather than a rounding error.
"""

from __future__ import annotations

import json
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "v2d.stage_timing.v1"


@dataclass
class ContainerRecord:
    """One ``docker run`` invocation measured from the host."""

    image: str
    module: str
    seconds: float
    started_at: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "module": self.module,
            "seconds": round(self.seconds, 3),
            "started_at": self.started_at,
            "status": self.status,
        }


@dataclass
class StageRecord:
    """One pipeline stage, bounded by consecutive ``_step`` markers."""

    index: int
    label: str
    skipped: bool
    started_at: float
    _started_perf: float
    seconds: float | None = None
    containers: list[ContainerRecord] = field(default_factory=list)

    @property
    def container_seconds(self) -> float:
        return sum(container.seconds for container in self.containers)

    @property
    def host_seconds(self) -> float:
        return max((self.seconds or 0.0) - self.container_seconds, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "skipped": self.skipped,
            "started_at": self.started_at,
            "seconds": round(self.seconds or 0.0, 3),
            "container_seconds": round(self.container_seconds, 3),
            "host_seconds": round(self.host_seconds, 3),
            "containers": [container.to_dict() for container in self.containers],
        }


class StageTimeline:
    """Accumulates stage and container timings for a single pipeline run."""

    def __init__(self, run_label: str, metadata: dict[str, Any] | None = None):
        self.run_label = run_label
        self.metadata = dict(metadata or {})
        self.stages: list[StageRecord] = []
        self._open: StageRecord | None = None
        self._started_at = time.time()
        self._started_perf = time.perf_counter()
        self._started_cpu = time.process_time()
        self.total_seconds: float | None = None
        self.host_cpu_seconds: float | None = None
        self.status = "running"

    def begin_stage(self, label: str, skipped: bool = False) -> None:
        """Close the stage that is open and open ``label`` in its place."""
        self._close_open_stage()
        record = StageRecord(
            index=len(self.stages),
            label=label,
            skipped=skipped,
            started_at=time.time(),
            _started_perf=time.perf_counter(),
        )
        self.stages.append(record)
        # A skipped stage did no work, so close it immediately rather than
        # charging it with the time until the next marker.
        if skipped:
            record.seconds = 0.0
        else:
            self._open = record

    def record_container(
        self, image: str, module: str, seconds: float, started_at: float, status: str
    ) -> None:
        """Attribute one ``docker run`` to whichever stage is currently open."""
        record = ContainerRecord(
            image=image,
            module=module,
            seconds=seconds,
            started_at=started_at,
            status=status,
        )
        if self._open is not None:
            self._open.containers.append(record)
        else:
            # A container launched outside any stage still belongs in the
            # timeline, so give it a synthetic stage of its own.
            self.begin_stage(f"(outside stage) {module}")
            assert self._open is not None
            self._open.containers.append(record)
            self._close_open_stage()

    def _close_open_stage(self) -> None:
        if self._open is None:
            return
        self._open.seconds = time.perf_counter() - self._open._started_perf
        self._open = None

    def finish(self, status: str = "complete") -> None:
        self._close_open_stage()
        self.total_seconds = time.perf_counter() - self._started_perf
        self.host_cpu_seconds = time.process_time() - self._started_cpu
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        measured = [stage for stage in self.stages if not stage.skipped]
        container_total = sum(stage.container_seconds for stage in self.stages)
        return {
            "schema": SCHEMA,
            "run_label": self.run_label,
            "status": self.status,
            "clock": {
                "durations": "time.perf_counter (monotonic wall clock)",
                "timestamps": "time.time (epoch seconds)",
                "host_cpu": "time.process_time (host orchestrator only)",
            },
            "started_at": self._started_at,
            "total_seconds": round(self.total_seconds or 0.0, 3),
            "host_cpu_seconds": round(self.host_cpu_seconds or 0.0, 3),
            "container_seconds": round(container_total, 3),
            "stage_count": len(self.stages),
            "stages_run": len(measured),
            "stages_skipped": len(self.stages) - len(measured),
            "environment": _environment(),
            "metadata": dict(sorted(self.metadata.items())),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def summary_table(self) -> str:
        """Render the timeline as a table sorted by descending wall time."""
        lines = [
            "",
            "=" * 78,
            f"  Stage timing -- {self.run_label} ({self.status})",
            "=" * 78,
            f"  {'stage':<46}{'wall':>9}{'docker':>9}{'host':>9}",
            "  " + "-" * 73,
        ]
        for stage in sorted(
            self.stages, key=lambda item: item.seconds or 0.0, reverse=True
        ):
            if stage.skipped:
                continue
            label = stage.label if len(stage.label) <= 45 else stage.label[:42] + "..."
            lines.append(
                f"  {label:<46}"
                f"{_fmt(stage.seconds or 0.0):>9}"
                f"{_fmt(stage.container_seconds):>9}"
                f"{_fmt(stage.host_seconds):>9}"
            )
        skipped = [stage for stage in self.stages if stage.skipped]
        total = self.total_seconds or 0.0
        container_total = sum(stage.container_seconds for stage in self.stages)
        lines.append("  " + "-" * 73)
        # The host column is the run total minus container time rather than the
        # sum of the stage host columns, so it also covers the work that falls
        # between stages and after the last one.
        lines.append(
            f"  {'TOTAL':<46}"
            f"{_fmt(total):>9}"
            f"{_fmt(container_total):>9}"
            f"{_fmt(max(total - container_total, 0.0)):>9}"
        )
        lines.append(f"  host CPU time (orchestrator): {_fmt(self.host_cpu_seconds or 0.0)}")
        if skipped:
            lines.append(f"  cached/skipped stages: {len(skipped)}")
        lines.append("=" * 78)
        lines.append("")
        return "\n".join(lines)

    def write(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return destination


def _fmt(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def _environment() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
    }


_ACTIVE: StageTimeline | None = None


def active_timeline() -> StageTimeline | None:
    """Return the timeline for the running pipeline, if any."""
    return _ACTIVE


def record_stage(label: str, skipped: bool = False) -> None:
    """Open ``label`` on the active timeline. A no-op when timing is off."""
    if _ACTIVE is not None:
        _ACTIVE.begin_stage(label, skipped=skipped)


def record_container(
    image: str, module: str, seconds: float, started_at: float, status: str
) -> None:
    """Attribute a ``docker run`` to the active timeline. No-op when timing is off."""
    if _ACTIVE is not None:
        _ACTIVE.record_container(image, module, seconds, started_at, status)


@contextmanager
def stage_timeline(
    run_label: str,
    report_path: str | os.PathLike[str] | None = None,
    metadata: dict[str, Any] | None = None,
    print_summary: bool = True,
):
    """Activate stage timing for the duration of the block.

    The report is written and printed on the way out even when the pipeline
    raises, so a run that dies halfway still reports where the time went.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        # Nested pipelines share the outer timeline rather than starting a
        # second one, which would split the run into two disjoint reports.
        yield _ACTIVE
        return
    timeline = StageTimeline(run_label, metadata=metadata)
    _ACTIVE = timeline
    try:
        yield timeline
    except BaseException:
        timeline.finish(status="failed")
        raise
    else:
        timeline.finish(status="complete")
    finally:
        _ACTIVE = None
        if print_summary:
            print(timeline.summary_table(), flush=True)
        if report_path is not None:
            try:
                written = timeline.write(report_path)
                print(f"  stage timing report: {written}", flush=True)
            except OSError as exc:  # Never fail a finished run over a report.
                print(f"  WARNING: could not write stage timing report: {exc}")
