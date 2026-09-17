from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable


def should_profile_cuda_step(start_step: int, step_count: int, global_step: int) -> bool:
    start_step = int(start_step)
    step_count = int(step_count)
    global_step = int(global_step)
    if start_step < 0 or step_count < 0:
        raise ValueError(f"CUDA profile step values must be nonnegative, got start={start_step}, count={step_count}")
    return step_count > 0 and start_step <= global_step < start_step + step_count


@contextmanager
def cuda_timing_section(timer: "CudaEventTimer | None", name: str):
    if timer is None:
        yield
        return
    with timer.section(name):
        yield


class CudaEventTimer:
    def __init__(self, enabled: bool, event_factory: Callable[[], Any] | None = None, synchronize: Callable[[], None] | None = None):
        self.enabled = bool(enabled)
        if self.enabled and (event_factory is None or synchronize is None):
            import torch

            event_factory = lambda: torch.cuda.Event(enable_timing=True)
            synchronize = torch.cuda.synchronize
        self.event_factory = event_factory
        self.synchronize = synchronize
        self.step = None
        self.events = []

    def begin_step(self, step: int) -> None:
        if not self.enabled:
            return
        if self.step is not None:
            raise RuntimeError(f"finish_step must be called before begin_step({step})")
        self.step = int(step)
        self.events = []

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if self.step is None:
            raise RuntimeError("begin_step must be called before recording CUDA sections")
        start = self.event_factory()
        end = self.event_factory()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((str(name), start, end))

    def finish_step(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self.step is None:
            raise RuntimeError("begin_step must be called before finish_step")
        self.synchronize()
        milliseconds = defaultdict(float)
        for name, start, end in self.events:
            milliseconds[name] += float(start.elapsed_time(end))
        result = {"step": self.step, "milliseconds": dict(milliseconds)}
        self.step = None
        self.events = []
        return result
