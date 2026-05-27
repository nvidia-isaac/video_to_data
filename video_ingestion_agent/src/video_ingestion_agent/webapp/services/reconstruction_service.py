# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Orchestrates the reconstruction chain via subprocess.

One chain: invoke `reconstruction_interface/ego_e2e/run_ego_e2e.py` once per
segment. That wrapper crosses into reconstruction's `.venv` and runs the
16-stage `run_v2d_ego_e2e.py` orchestrator. The service parses the
orchestrator's `[run ] <label>` / `[skip] <label>` stdout markers to drive
the status bar; caching is delegated to the orchestrator's `_step` /
`_has_files` short-circuit logic.

No reconstruction Python packages are imported here — the heavy lifting
stays in Docker (per-stage v2d_* containers spawned by the orchestrator)
or in reconstruction's own `.venv` (the orchestrator interpreter itself).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from video_ingestion_agent.webapp.models.reconstruction import (
    STAGE_MARKER,
    STAGES,
    ReconstructionRequest,
    ReconstructionResult,
    StageEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionConfig:
    """Where the weights and Docker images live."""

    out_root: Path = Path("outputs/webapp/reconstruction")

    # Weight directories (host paths).
    moge_weights: Path | None = None
    grounding_dino_weights: Path | None = None
    sam2_weights: Path | None = None
    sam3d_weights: Path | None = None
    foundation_pose_weights: Path | None = None
    # Hand reconstruction needs MANO_RIGHT.pkl + BMC/.
    hand_reconstruction_weights: Path | None = None
    mano_weights: Path | None = None  # defaults to hand_reconstruction_weights if unset

    # Container image tags (defaults match reconstruction/scripts/build_containers.sh).
    moge_image: str = "v2d_moge:latest"
    grounding_dino_image: str = "v2d_grounding_dino:latest"
    sam2_image: str = "v2d_sam2:latest"
    sam3d_image: str = "v2d_sam3d:latest"
    mesh_image: str = "v2d_mesh:latest"
    foundation_pose_image: str = "v2d_foundation_pose:latest"
    hand_alignment_image: str = "v2d_hand_alignment:latest"

    # Python interpreter to invoke the wrapper script with. Defaults to the
    # one running the webapp so we don't accidentally pick a different env.
    python: str = sys.executable

    # Bridge into reconstruction: paths to its `.venv` + source tree. The
    # wrapper subprocesses into reconstruction's lightweight orchestration
    # venv (no torch / CUDA) to invoke run_v2d_ego_e2e.py.
    reconstruction_python: Path | None = None
    reconstruction_root: Path | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ReconstructionConfig:
        """Build a config from a YAML-loaded dict (the `reconstruction:` block)."""
        weights = data.get("weights", {}) or {}
        images = data.get("images", {}) or {}

        def _path(key: str) -> Path | None:
            v = weights.get(key)
            return Path(v).expanduser() if v else None

        recon_python = data.get("reconstruction_python")
        recon_root = data.get("reconstruction_root")
        return cls(
            out_root=Path(data.get("out_root", "outputs/webapp/reconstruction")).expanduser(),
            moge_weights=_path("moge"),
            grounding_dino_weights=_path("grounding_dino"),
            sam2_weights=_path("sam2"),
            sam3d_weights=_path("sam3d"),
            foundation_pose_weights=_path("foundation_pose"),
            hand_reconstruction_weights=_path("hand_reconstruction"),
            mano_weights=_path("mano"),
            moge_image=images.get("moge", "v2d_moge:latest"),
            grounding_dino_image=images.get("grounding_dino", "v2d_grounding_dino:latest"),
            sam2_image=images.get("sam2", "v2d_sam2:latest"),
            sam3d_image=images.get("sam3d", "v2d_sam3d:latest"),
            mesh_image=images.get("mesh", "v2d_mesh:latest"),
            foundation_pose_image=images.get("foundation_pose", "v2d_foundation_pose:latest"),
            hand_alignment_image=images.get("hand_alignment", "v2d_hand_alignment:latest"),
            reconstruction_python=Path(recon_python).expanduser() if recon_python else None,
            reconstruction_root=Path(recon_root).expanduser() if recon_root else None,
        )

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty = ready to run)."""
        problems: list[str] = []
        for name, p in (
            ("moge", self.moge_weights),
            ("grounding_dino", self.grounding_dino_weights),
            ("sam2", self.sam2_weights),
            ("sam3d", self.sam3d_weights),
            ("foundation_pose", self.foundation_pose_weights),
            ("hand_reconstruction", self.hand_reconstruction_weights),
        ):
            if p is None:
                problems.append(f"{name} weights path not configured")
            elif not p.is_dir():
                problems.append(f"{name} weights dir missing: {p}")
        for image in (
            self.moge_image,
            self.grounding_dino_image,
            self.sam2_image,
            self.sam3d_image,
            self.mesh_image,
            self.foundation_pose_image,
            self.hand_alignment_image,
        ):
            try:
                subprocess.run(
                    ["docker", "image", "inspect", image],
                    check=True,
                    capture_output=True,
                )
            except FileNotFoundError:
                problems.append("docker CLI not on PATH")
                break
            except subprocess.CalledProcessError:
                problems.append(f"docker image not built: {image}")
        if self.reconstruction_python is None:
            problems.append("reconstruction_python not configured")
        elif not self.reconstruction_python.is_file():
            problems.append(f"reconstruction_python not found: {self.reconstruction_python}")
        if self.reconstruction_root is None:
            problems.append("reconstruction_root not configured")
        elif not self.reconstruction_root.is_dir():
            problems.append(f"reconstruction_root not found: {self.reconstruction_root}")
        return problems


_STAGE_MODULE = "video_ingestion_agent.reconstruction_interface.ego_e2e.run_ego_e2e"

# Match the e2e orchestrator's `_step()` output format:
#   `  [run ] <label>` while running
#   `  [skip] <label>` when artifact already exists
_MARKER_RE = re.compile(r"^\s*\[(run |skip)\]\s+(.+?)\s*$")


@dataclass
class ReconstructionService:
    """Run the chain on one segment at a time, yielding StageEvent updates."""

    config: ReconstructionConfig
    log_lines_per_stage: int = 200

    # ------------------------------------------------------------------ public

    def run(self, request: ReconstructionRequest) -> Iterator[StageEvent]:
        """Run the full chain, yielding one StageEvent per major status change.

        Caching: the upstream orchestrator's `_step` short-circuit emits
        `[skip] <label>` whenever a stage's primary artifact is already on
        disk; the parser below converts those to immediate `ok` events.

        Yields:
            - ('<stage>', 'running', message=…)         on every captured log line
            - ('<stage>', 'ok'|'err', elapsed_s=…)      when the subprocess exits

        The full subprocess stdout is written verbatim to
        `<seg_dir>/ego_e2e.log` so container errors that fall between the
        UI's throttled events (every 10th line after the first 50) survive
        for offline diagnosis.
        """
        seg_dir = (self.config.out_root / request.segment_id).resolve()
        seg_dir.mkdir(parents=True, exist_ok=True)

        segments_jsonl = self._write_segment_jsonl(request)

        try:
            yield from self._run(request, segments_jsonl, seg_dir)
        finally:
            try:
                segments_jsonl.unlink()
            except FileNotFoundError:
                pass

    def collect_result(self, request: ReconstructionRequest) -> ReconstructionResult:
        """Build a snapshot of what's currently on disk for this segment."""
        seg_dir = (self.config.out_root / request.segment_id).resolve()
        return ReconstructionResult(request=request, seg_dir=seg_dir)

    # ------------------------------------------------------------------ subprocess

    def _run(
        self,
        request: ReconstructionRequest,
        segments_jsonl: Path,
        seg_dir: Path,
    ) -> Iterator[StageEvent]:
        """Spawn the wrapper (which calls reconstruction's run_v2d_ego_e2e.py
        orchestrator) and parse its stdout to drive a 16-stage status bar.

        The orchestrator emits one of these per `_step()` call:
            ``  [run ] <label>``    — stage starting
            ``  [skip] <label>``    — primary artifact already on disk
        We map ``<label>`` to a stage_id via STAGE_MARKER and emit
        StageEvents accordingly. Non-marker lines flow through as throttled
        ``running`` messages so the log viewer stays useful.

        Caching is delegated entirely to the orchestrator; we don't pre-check.
        """
        cmd = self._build_cmd(request, segments_jsonl)
        logger.info("[reconstruction] %s", shlex.join(map(str, cmd)))
        yield StageEvent(
            stage=STAGES[0],
            status="running",
            message=f"$ {shlex.join(map(str, cmd[:6]))} …",
        )

        log_path = seg_dir / "ego_e2e.log"
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                cwd=os.getcwd(),
            )
        except FileNotFoundError as e:
            yield StageEvent(stage=STAGES[0], status="err", message=str(e))
            return

        line_count = 0
        last_line = ""
        current_stage: str | None = None
        assert proc.stdout is not None
        with log_path.open("w") as logf:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                line_count += 1
                last_line = line.rstrip()

                marker = _MARKER_RE.match(line)
                if marker:
                    action = marker.group(1).strip()  # "run" or "skip"
                    label = marker.group(2)
                    matched = next(
                        (sid for prefix, sid in STAGE_MARKER if label.startswith(prefix)),
                        None,
                    )
                    if matched is not None:
                        # Close out the previous stage (if any was running).
                        if current_stage is not None and current_stage != matched:
                            yield StageEvent(stage=current_stage, status="ok", message="")
                        if action == "skip":
                            yield StageEvent(
                                stage=matched,
                                status="ok",
                                elapsed_s=0.0,
                                message=f"(cached — orchestrator skipped: {label})",
                            )
                            current_stage = None
                        else:  # "run"
                            current_stage = matched
                            yield StageEvent(stage=matched, status="running", message=last_line)
                        continue

                # Non-marker line: pass through to the log viewer (throttled).
                # The tab-side guard refuses to downgrade an already-"ok"
                # stage back to "running" via these passthrough events, so
                # attributing them to STAGES[0] when no stage is actively
                # running is safe (the bar stays honest, only the log_buf
                # line prefix may be slightly off).
                if line_count <= 50 or line_count % 10 == 0:
                    yield StageEvent(
                        stage=current_stage or STAGES[0],
                        status="running",
                        message=last_line,
                    )
        proc.wait()
        elapsed = time.monotonic() - start

        if proc.returncode == 0:
            # Close out the last running stage (if any).
            if current_stage is not None:
                yield StageEvent(
                    stage=current_stage,
                    status="ok",
                    elapsed_s=elapsed,
                    message=last_line,
                )
        else:
            tail = self._tail(log_path, lines=30)
            yield StageEvent(
                stage=current_stage or STAGES[0],
                status="err",
                elapsed_s=elapsed,
                message=(
                    f"exit {proc.returncode} — full log: {log_path}\n--- last 30 lines ---\n{tail}"
                ),
            )

    @staticmethod
    def _tail(path: Path, lines: int = 30) -> str:
        try:
            with path.open() as f:
                buf = f.readlines()
        except FileNotFoundError:
            return "(log file missing)"
        return "".join(buf[-lines:])

    def _build_cmd(
        self,
        request: ReconstructionRequest,
        segments_jsonl: Path,
    ) -> list[str]:
        """Construct the `python -m …ego_e2e.run_ego_e2e` argv.

        The wrapper itself orchestrates 16 docker subprocesses per segment;
        we drive it once per request.
        """
        out_root = self.config.out_root.resolve()
        return [
            self.config.python,
            "-m",
            _STAGE_MODULE,
            "--segments",
            str(segments_jsonl),
            "--out",
            str(out_root),
            "--reconstruction-python",
            str(self.config.reconstruction_python),
            "--reconstruction-root",
            str(self.config.reconstruction_root),
            "--moge-weights",
            str(self.config.moge_weights),
            "--grounding-dino-weights",
            str(self.config.grounding_dino_weights),
            "--sam2-weights",
            str(self.config.sam2_weights),
            "--sam3d-weights",
            str(self.config.sam3d_weights),
            "--foundation-pose-weights",
            str(self.config.foundation_pose_weights),
            "--hand-reconstruction-weights",
            str(self.config.hand_reconstruction_weights),
            "--depth-source",
            request.depth_source,
            "--ref-frame",
            str(request.ref_frame),
        ]

    def _write_segment_jsonl(self, request: ReconstructionRequest) -> Path:
        """Persist the request as a one-line clips_final.jsonl-style file."""
        fd, name = tempfile.mkstemp(prefix=f"recon_{request.segment_id}_", suffix=".jsonl")
        path = Path(name)
        with os.fdopen(fd, "w") as f:
            f.write(
                json.dumps(
                    {
                        "clip_id": request.segment_id,
                        "video_path": str(request.video_path),
                        "start_t": request.start_t,
                        "end_t": request.end_t,
                        "object": request.object_label,
                        "action": request.action_label,
                        "description": request.description,
                    }
                )
                + "\n"
            )
        return path
