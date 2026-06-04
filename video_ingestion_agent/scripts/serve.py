#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
vLLM server manager for video_ingestion_agent.

Start, stop, and monitor a vLLM server that serves the VLM model
for the video segmentation pipeline.

Usage:
    python scripts/serve.py                        # Start server
    python scripts/serve.py -c custom_config.yaml  # Start with custom config
    python scripts/serve.py --stop                 # Stop the running server
    python scripts/serve.py --status               # Check if server is running
    python scripts/serve.py --logs                 # Tail the server logs
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VIDEO_INGESTION_AGENT_DIR = Path.home() / ".video_ingestion_agent"
PID_FILE = VIDEO_INGESTION_AGENT_DIR / "vllm.pid"
LOG_FILE = VIDEO_INGESTION_AGENT_DIR / "vllm.log"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_server(api_url: str, timeout: float = 3.0) -> bool:
    """Check if the vLLM server is reachable.

    Args:
        api_url: The vLLM base URL (e.g. "http://localhost:8000/v1").
        timeout: Request timeout in seconds.

    Returns:
        True if server responds, False otherwise.
    """
    import requests

    # /v1 -> /health
    parsed = urlparse(api_url)
    health_url = f"{parsed.scheme}://{parsed.netloc}/health"

    try:
        resp = requests.get(health_url, timeout=timeout)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------


def start_server(
    model_name: str,
    api_url: str = "http://localhost:8000/v1",
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
) -> int:
    """Start a vLLM server as a background process.

    Args:
        model_name: HuggingFace model name to serve.
        api_url: Target URL. Port is extracted from this.
        max_model_len: Maximum model context length.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use (0.0-1.0).

    Returns:
        PID of the vLLM process.
    """
    # Check if already running
    if check_server(api_url, timeout=2.0):
        print(f"vLLM server is already running at {api_url}")
        return _read_pid() or 0

    # Extract port from URL
    parsed = urlparse(api_url)
    port = parsed.port or 8000

    is_cosmos3 = "cosmos3" in model_name.lower()

    # Build command
    cmd = [
        "vllm",
        "serve",
        model_name,
        "--port",
        str(port),
        "--allowed-local-media-path",
        "/",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--media-io-kwargs",
        '{"video": {"num_frames": -1}}',
    ]

    if is_cosmos3:
        # Load only the Reasoner (VLM) tower of the Cosmos3 Omni model
        # (requires the vllm-cosmos3 plugin).
        cmd.extend(
            [
                "--hf-overrides",
                '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}',
                "--mm-encoder-tp-mode",
                "data",
                "--async-scheduling",
            ]
        )
    else:
        # Qwen3-VL processor knobs (invalid for Cosmos3).
        cmd.extend(
            [
                "--mm-processor-kwargs",
                '{"min_pixels": 262144, "max_pixels": 8388608}',
            ]
        )

    if tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

    # Ensure state directory exists
    VIDEO_INGESTION_AGENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting vLLM server...")
    print(f"  Model:  {model_name}")
    print(f"  TP:     {tensor_parallel_size} GPU(s)")
    print(f"  GPU mem: {gpu_memory_utilization:.0%}")
    print(f"  URL:    http://localhost:{port}/v1")
    print(f"  Logs:   {LOG_FILE}")

    # Launch as background process
    log_fh = open(LOG_FILE, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from terminal
    )

    # Save PID
    PID_FILE.write_text(str(proc.pid))

    # Wait for server to become healthy
    print(f"  PID:    {proc.pid}")
    print("  Waiting for server to load model (this may take 1-2 minutes)...")

    start_time = time.time()
    timeout = 600  # 10 minutes max (model download + load can be slow)
    poll_interval = 5

    while time.time() - start_time < timeout:
        # Check if process died
        ret = proc.poll()
        if ret is not None:
            print(f"\nERROR: vLLM process exited with code {ret}")
            print(f"Check logs: {LOG_FILE}")
            _cleanup_pid()
            sys.exit(1)

        if check_server(api_url, timeout=2.0):
            elapsed = time.time() - start_time
            print(f"\nvLLM server is ready! (took {elapsed:.0f}s)")
            print(f"  URL:  http://localhost:{port}/v1")
            return proc.pid

        # Progress indicator
        elapsed = int(time.time() - start_time)
        print(f"\r  [{elapsed}s] Loading model...", end="", flush=True)
        time.sleep(poll_interval)

    # Timeout
    print(f"\nERROR: Server did not become healthy within {timeout}s")
    print(f"Check logs: {LOG_FILE}")
    proc.terminate()
    _cleanup_pid()
    sys.exit(1)


def stop_server() -> None:
    """Stop the running vLLM server."""
    pid = _read_pid()

    if pid is None:
        print("No vLLM server PID file found.")
        print(f"  (looked for {PID_FILE})")
        return

    # Check if process is actually running
    try:
        os.kill(pid, 0)  # signal 0 = check existence
    except ProcessLookupError:
        print(f"vLLM process (PID {pid}) is not running (stale PID file).")
        _cleanup_pid()
        return
    except PermissionError:
        pass  # process exists but we can't signal it (shouldn't happen)

    print(f"Stopping vLLM server (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)

        # Wait for graceful shutdown
        for _ in range(15):
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except ProcessLookupError:
                break
        else:
            # Force kill if still alive
            print("  Sending SIGKILL...")
            os.kill(pid, signal.SIGKILL)

        print("vLLM server stopped.")
    except ProcessLookupError:
        print("vLLM server already stopped.")

    _cleanup_pid()


def show_status(api_url: str) -> None:
    """Print server status."""
    pid = _read_pid()
    alive = check_server(api_url, timeout=3.0)

    if alive:
        print(f"vLLM server is RUNNING at {api_url}")
        if pid:
            print(f"  PID: {pid}")
    elif pid:
        # PID file exists but server not responding
        try:
            os.kill(pid, 0)
            print(f"vLLM process is running (PID {pid}) but not yet healthy.")
            print("  It may still be loading the model. Check logs:")
            print(f"    tail -f {LOG_FILE}")
        except ProcessLookupError:
            print("vLLM server is NOT running (stale PID file).")
            _cleanup_pid()
    else:
        print("vLLM server is NOT running.")
        print("  Start it with: python scripts/serve.py -c configs/ingestion.yaml")


def tail_logs(lines: int = 50) -> None:
    """Print recent server logs."""
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}")
        return

    print(f"--- {LOG_FILE} (last {lines} lines) ---")
    with open(LOG_FILE) as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line, end="")
    print("\n--- end of logs ---")
    print(f"For live logs: tail -f {LOG_FILE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pid() -> int | None:
    """Read PID from file, return None if not found."""
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def _cleanup_pid() -> None:
    """Remove PID file."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Manage the vLLM server for video_ingestion_agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/serve.py -c configs/ingestion.yaml   # Start server
  python scripts/serve.py --status                    # Check status
  python scripts/serve.py --stop                      # Stop server
  python scripts/serve.py --logs                      # View server logs
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        default="configs/ingestion.yaml",
        help="Pipeline config YAML (default: configs/ingestion.yaml)",
    )
    parser.add_argument("--stop", action="store_true", help="Stop the running server")
    parser.add_argument("--status", action="store_true", help="Check server status")
    parser.add_argument("--logs", action="store_true", help="Show recent server logs")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model context length (default: 32768)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size (number of GPUs). Overrides config vllm_tp_size.",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help=(
            "Fraction of GPU memory for vLLM (0.0-1.0). "
            "Overrides config vllm_gpu_memory_utilization. "
            "Default 0.8 reserves ~20%% for embeddings."
        ),
    )

    args = parser.parse_args()

    # Load config for model name and URL
    from video_ingestion_agent.ingestion.config import load_config

    config = load_config(args.config)
    model_name = config.models.vlm_model
    api_url = config.models.vllm_url
    tp_size = args.tp if args.tp is not None else config.models.vllm_tp_size
    gpu_mem = (
        args.gpu_mem if args.gpu_mem is not None else config.models.vllm_gpu_memory_utilization
    )
    vlm_backend = config.models.vlm_backend

    if args.stop:
        stop_server()
    elif args.status:
        show_status(api_url)
    elif args.logs:
        tail_logs()
    else:
        # Only start vLLM when the backend is "vllm"
        if vlm_backend != "vllm":
            print(
                f"vlm_backend is '{vlm_backend}' (not 'vllm') -- no vLLM server needed. Skipping."
            )
            sys.exit(0)

        start_server(
            model_name=model_name,
            api_url=api_url,
            max_model_len=args.max_model_len,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem,
        )


if __name__ == "__main__":
    main()
