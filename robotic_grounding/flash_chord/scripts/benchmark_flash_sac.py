# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run FlashSAC with synchronized throughput windows after warmup.

All unrecognized arguments are passed to the normal Hydra training entry point.
This measures training rather than changing its algorithm or default settings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import runpy
import statistics
import sys
import time

import jax
import warp as wp

from flash_chord.training.flash_sac import runner


def main() -> None:
    """Run training, synchronize each measurement window, and print JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-window", type=int, default=16, help="collections per measured window")
    parser.add_argument("--benchmark-warmup", type=int, default=16, help="collections before timing")
    args, hydra_args = parser.parse_known_args()
    if args.benchmark_window <= 0 or args.benchmark_warmup <= 0:
        parser.error("benchmark window and warmup must both be positive")

    original_compile = runner._compile_executables
    windows: list[dict[str, float | int]] = []

    def compile_with_timing(*compile_args, **compile_kwargs):
        executables = original_compile(*compile_args, **compile_kwargs)
        original_consume = executables.consume_transition
        collections = 0
        window_start = None
        previous_updates = None

        def timed_consume(*call_args, **call_kwargs):
            nonlocal collections, window_start, previous_updates
            result = original_consume(*call_args, **call_kwargs)
            collections += 1
            boundary = collections == args.benchmark_warmup or (
                collections > args.benchmark_warmup
                and (collections - args.benchmark_warmup) % args.benchmark_window == 0
            )
            if boundary:
                jax.block_until_ready(result[0].global_update_step)
                wp.synchronize_device()
                now = time.perf_counter()
                update_step = int(jax.device_get(result[0].global_update_step))
                if window_start is not None:
                    elapsed = now - window_start
                    world_count = int(call_args[1].action.shape[0])
                    transitions = args.benchmark_window * world_count
                    windows.append(
                        {
                            "world_count": world_count,
                            "environment_steps": transitions,
                            "updates": update_step - previous_updates,
                            "seconds": elapsed,
                            "environment_steps_per_second": transitions / elapsed,
                        }
                    )
                window_start = now
                previous_updates = update_step
            return result

        return replace(executables, consume_transition=timed_consume)

    runner._compile_executables = compile_with_timing
    script = Path(__file__).with_name("train_flash_sac.py")
    sys.argv = [str(script), *hydra_args]
    runpy.run_path(str(script), run_name="__main__")
    if not windows:
        raise RuntimeError("training ended before a complete benchmark window; increase total_environment_steps")
    print(
        "FLASHSAC_BENCHMARK "
        + json.dumps(
            {
                "warmup_collections": args.benchmark_warmup,
                "window_collections": args.benchmark_window,
                "windows": windows,
                "median_environment_steps_per_second": statistics.median(
                    window["environment_steps_per_second"] for window in windows
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
