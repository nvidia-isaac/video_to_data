#!/usr/bin/env python3
"""Download visualizer recordings from OSMO to local datasets dir.

Syncs {name}_html/ trees (*.viser, *.mp4, viser-client/) from the OSMO
isaac bucket using `osmo dataset download`. Multiple datasets can be
downloaded in parallel with --jobs (use MAX to download all at once).
Ctrl+C terminates all running downloads immediately.

Prerequisites:
  - osmo CLI on PATH and authenticated

Usage:
  python robotic_grounding/visualizer/sync_visualizer_data.py
  python robotic_grounding/visualizer/sync_visualizer_data.py --dataset arctic
  python robotic_grounding/visualizer/sync_visualizer_data.py --dataset arctic h2o
  python robotic_grounding/visualizer/sync_visualizer_data.py --jobs MAX
  python robotic_grounding/visualizer/sync_visualizer_data.py --dry-run
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS = ("arctic", "h2o", "grab", "taco", "dexycb", "hot3d")

LOCAL_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# Rich color per dataset (same order as DATASETS)
_RICH_COLORS = ("cyan", "green", "yellow", "magenta", "blue", "red")

# Regex to parse a tqdm progress line emitted by osmo
# Matches: "  17%|███  | 181M/1.06G [00:05<00:22, 42.5MB/s, ...]"
_TQDM_RE = re.compile(
    r"^\s*(\d+)%\|"               # percentage
    r".*?\|\s*"
    r"([\d.]+\s*\S+)"             # bytes done
    r"/"
    r"([\d.]+\s*\S+)"             # bytes total
    r"\s*\["
    r"[^,<]+"                     # elapsed
    r"<?[^,]*"                    # eta
    r",?\s*([\d.]+\s*\S+/s)?"    # speed (optional)
)

# Shutdown flag + registry of active subprocesses for Ctrl+C cleanup
_shutdown = threading.Event()
_active_procs: list[subprocess.Popen] = []
_procs_lock = threading.Lock()


def _terminate_all() -> None:
    """Kill every active osmo process group (osmo spawns ~64 child workers)."""
    with _procs_lock:
        for proc in _active_procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rich_color(name: str) -> str:
    try:
        return _RICH_COLORS[list(DATASETS).index(name)]
    except ValueError:
        return "white"


def _osmo_dataset(name: str) -> str:
    return f"isaac/v2d_{name}_retarget_exp_200"


def _local_dir(name: str) -> Path:
    return LOCAL_DATASETS_DIR / f"v2d_{name}_retarget_exp_200"


def _parse_tqdm(line: str) -> tuple[int, str, str, str] | None:
    """Parse a tqdm progress line.

    Returns (pct, done, total, speed) or None if the line isn't a tqdm bar.
    """
    m = _TQDM_RE.match(line)
    if not m:
        return None
    pct = int(m.group(1))
    done = m.group(2).strip()
    total = m.group(3).strip()
    speed = (m.group(4) or "").strip()
    return pct, done, total, speed


# ---------------------------------------------------------------------------
# Streaming output → rich progress
# ---------------------------------------------------------------------------
def _stream(
    proc: subprocess.Popen,
    name: str,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Read proc stdout+stderr, update the rich task on tqdm lines,
    and print other lines (errors, info) above the progress bars.
    """
    color = _rich_color(name)
    buf = b""

    while True:
        chunk = proc.stdout.read(256)
        if not chunk:
            break
        buf += chunk

        while True:
            ni = buf.find(b"\n")
            ri = buf.find(b"\r")
            if ni == -1 and ri == -1:
                break
            if ni == -1 or (ri != -1 and ri < ni):
                raw, buf = buf[:ri], buf[ri + 1:]
            else:
                raw, buf = buf[:ni], buf[ni + 1:]

            line = raw.decode(errors="replace").strip()
            if not line:
                continue

            parsed = _parse_tqdm(line)
            if parsed:
                pct, done, total, speed = parsed
                label = f"[{color}]{name:<7}[/{color}]"
                info = f"[dim]{done}/{total}"
                if speed:
                    info += f" @ {speed}"
                info += "[/dim]"
                progress.update(task_id, completed=pct, description=f"{label} {info}")
            else:
                progress.console.print(f"[{color}]\\[{name}][/{color}] {line}")

    if buf.strip():
        line = buf.decode(errors="replace").strip()
        progress.console.print(f"[{_rich_color(name)}]\\[{name}][/{_rich_color(name)}] {line}")


# ---------------------------------------------------------------------------
# Core sync
# ---------------------------------------------------------------------------
def sync(
    name: str,
    pattern: str | None,
    dry_run: bool,
    progress: Progress,
) -> int:
    """Download {name}_html/ from OSMO. Returns osmo exit code."""
    if _shutdown.is_set():
        progress.console.print(f"[dim]\\[{name}] skipped (interrupted)[/dim]")
        return 1

    dest = _local_dir(name)
    regex = rf"^{name}_html/.*{pattern}" if pattern else rf"^{name}_html/"

    # osmo ALWAYS creates a DATASET_NAME/ subdirectory inside whatever path it
    # receives. So pass LOCAL_DATASETS_DIR (the parent); osmo then creates
    # v2d_{name}_retarget_exp_200/ inside it — which is exactly dest.
    # Delete dest first so osmo creates it fresh without nesting inside itself.
    cmd = [
        "osmo", "dataset", "download",
        _osmo_dataset(name),
        str(LOCAL_DATASETS_DIR),
        "--regex", regex,
    ]

    color = _rich_color(name)
    progress.console.print(
        f"[{color}]\\[{name}][/{color}] "
        f"{_osmo_dataset(name)} [dim]→[/dim] {dest}"
    )

    if dry_run:
        progress.console.print(
            f"[{color}]\\[{name}][/{color}] [dim][dry-run] {' '.join(cmd)}[/dim]"
        )
        return 0

    LOCAL_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)  # remove so osmo creates dest/ fresh, not dest/dest/

    task_id = progress.add_task(
        f"[{color}]{name:<7}[/{color}] [dim]starting...[/dim]",
        total=100,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,   # own process group → killpg kills all workers
    )
    with _procs_lock:
        _active_procs.append(proc)
    try:
        _stream(proc, name, progress, task_id)
        rc = proc.wait()
    finally:
        with _procs_lock:
            try:
                _active_procs.remove(proc)
            except ValueError:
                pass

    if rc == 0:
        progress.update(task_id, completed=100,
                        description=f"[{color}]{name:<7}[/{color}] [green]done[/green]")
    elif not _shutdown.is_set():
        progress.update(task_id,
                        description=f"[{color}]{name:<7}[/{color}] [red]failed (code {rc})[/red]")
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download visualizer recordings from OSMO to local datasets dir.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=[*DATASETS, "all"],
        default=["all"],
        help="Dataset(s) to sync, or 'all' (default).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Extra regex to narrow files within {name}_html/ (e.g. 'arctic_s01').",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=str,
        default="1",
        metavar="N|MAX",
        help="Parallel downloads: integer or MAX to download all datasets at once.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the osmo command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = list(DATASETS) if "all" in args.dataset else args.dataset

    jobs_str = args.jobs.upper()
    n_jobs = len(datasets) if jobs_str == "MAX" else int(args.jobs)

    progress = Progress(
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        refresh_per_second=10,
        transient=False,
    )

    # ---- sequential path -----------------------------------------------
    if n_jobs == 1:
        try:
            with progress:
                for name in datasets:
                    if _shutdown.is_set():
                        break
                    rc = sync(name, args.pattern, args.dry_run, progress)
                    if rc != 0 and not _shutdown.is_set():
                        sys.exit(rc)
        except KeyboardInterrupt:
            print("\nInterrupted — terminating download.")
            _shutdown.set()
            _terminate_all()
            os._exit(130)
        return

    # ---- parallel path -------------------------------------------------
    failed: list[str] = []
    pool = ThreadPoolExecutor(max_workers=n_jobs)
    futures = {
        pool.submit(sync, name, args.pattern, args.dry_run, progress): name
        for name in datasets
    }
    try:
        with progress:
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    rc = fut.result()
                except Exception as exc:
                    progress.console.print(f"[red]\\[{name}] EXCEPTION: {exc}[/red]")
                    failed.append(name)
                else:
                    if rc != 0:
                        failed.append(name)
    except KeyboardInterrupt:
        print("\nInterrupted — terminating all downloads.")
        _shutdown.set()
        _terminate_all()
        pool.shutdown(wait=False, cancel_futures=True)
        os._exit(130)

    pool.shutdown(wait=True)

    if failed:
        print(f"Failed datasets: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
