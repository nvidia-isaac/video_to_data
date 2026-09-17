import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


PENDING_VALIDATION_FILENAME = ".pending_validation.json"


def parse_slurm_duration_seconds(value: str) -> int:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid Slurm duration: {value!r}")
    if value.upper() == "UNLIMITED":
        raise ValueError("Slurm duration must be finite")
    day_parts = value.split("-")
    if len(day_parts) > 2 or any(not part for part in day_parts):
        raise ValueError(f"Invalid Slurm duration: {value!r}")
    days = 0
    time_text = day_parts[-1]
    if len(day_parts) == 2:
        if not day_parts[0].isdigit():
            raise ValueError(f"Invalid Slurm duration: {value!r}")
        days = int(day_parts[0])
    parts = time_text.split(":")
    if len(parts) not in (2, 3) or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid Slurm duration: {value!r}")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    else:
        hours = 0
        minutes, seconds = map(int, parts)
    if minutes >= 60 or seconds >= 60 or (days and hours >= 24):
        raise ValueError(f"Invalid Slurm duration: {value!r}")
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError(f"Slurm duration must be positive: {value!r}")
    return total


@dataclass(frozen=True)
class SegmentRuntimeController:
    start_time_epoch: float
    time_limit_seconds: int
    checkpoint_margin_seconds: int
    check_interval_steps: int
    checkpoint_request_path: Optional[Path]

    @classmethod
    def from_environment(cls, env: Optional[Mapping[str, str]] = None):
        env = os.environ if env is None else env
        start_text = env.get("MHR_SEGMENT_START_TIME_EPOCH")
        limit_text = env.get("MHR_SEGMENT_TIME_LIMIT")
        if start_text is None and limit_text is None:
            return None
        if start_text is None or limit_text is None:
            raise ValueError("MHR_SEGMENT_START_TIME_EPOCH and MHR_SEGMENT_TIME_LIMIT must be provided together")
        start_time_epoch = float(start_text)
        if not math.isfinite(start_time_epoch) or start_time_epoch <= 0:
            raise ValueError(f"Invalid MHR_SEGMENT_START_TIME_EPOCH: {start_text!r}")
        time_limit_seconds = parse_slurm_duration_seconds(limit_text)
        margin = int(env.get("MHR_SEGMENT_CHECKPOINT_MARGIN_SECONDS", "360"))
        interval = int(env.get("MHR_SEGMENT_RUNTIME_CHECK_INTERVAL_STEPS", "10"))
        if margin <= 0 or margin >= time_limit_seconds:
            raise ValueError(f"Checkpoint margin must be in (0, {time_limit_seconds}), got {margin}")
        if interval <= 0:
            raise ValueError(f"Runtime check interval must be positive, got {interval}")
        request_text = env.get("MHR_SEGMENT_CHECKPOINT_REQUEST_PATH")
        request_path = None if not request_text else Path(request_text)
        return cls(start_time_epoch=start_time_epoch, time_limit_seconds=time_limit_seconds, checkpoint_margin_seconds=margin, check_interval_steps=interval, checkpoint_request_path=request_path)

    def should_check(self, global_step: int, force: bool = False) -> bool:
        return force or global_step % self.check_interval_steps == 0

    def elapsed_seconds(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, float(now) - self.start_time_epoch)

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        return self.time_limit_seconds - self.elapsed_seconds(now)

    def local_reason(self, global_step: int, now: Optional[float] = None, force: bool = False) -> Optional[str]:
        if not self.should_check(global_step, force=force):
            return None
        if self.checkpoint_request_path is not None and self.checkpoint_request_path.is_file():
            return "usr1"
        if self.remaining_seconds(now) <= self.checkpoint_margin_seconds:
            return "deadline"
        return None

    def metadata(self, global_step: int, reason: str, now: Optional[float] = None) -> dict:
        elapsed = self.elapsed_seconds(now)
        return {"reason": reason, "global_step": int(global_step), "start_time_epoch": self.start_time_epoch, "time_limit_seconds": self.time_limit_seconds, "checkpoint_margin_seconds": self.checkpoint_margin_seconds, "elapsed_seconds": elapsed, "remaining_seconds": self.time_limit_seconds - elapsed}

    def write_checkpoint_completion(self, metadata: dict) -> Path:
        if self.checkpoint_request_path is None:
            raise RuntimeError("Runtime checkpoint completion requires MHR_SEGMENT_CHECKPOINT_REQUEST_PATH")
        return _write_atomic_json(Path(f"{self.checkpoint_request_path}.completed"), metadata)


def pending_validation_path(run_dir: Path) -> Path:
    return Path(run_dir) / PENDING_VALIDATION_FILENAME


def _write_atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temp_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temp_path, path)
    return path


def write_pending_validation(run_dir: Path, step: int, reason: str) -> Path:
    if type(step) is not int or step < 0:
        raise ValueError(f"Pending validation step must be a non-negative integer, got {step!r}")
    if not isinstance(reason, str) or not reason:
        raise ValueError("Pending validation reason must be a non-empty string")
    return _write_atomic_json(pending_validation_path(run_dir), {"reason": reason, "step": step})


def pending_validation_step(run_dir: Path) -> Optional[int]:
    path = pending_validation_path(run_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    step = payload.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(f"Pending validation marker step must be a non-negative integer: {path}")
    return step


def clear_pending_validation(run_dir: Path):
    pending_validation_path(run_dir).unlink(missing_ok=True)
