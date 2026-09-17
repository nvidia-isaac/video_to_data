# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch native MHR CoCoNet training inside the CARI4D container."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


SOURCE_ROOT = Path(__file__).resolve().parent / "cari4d"
DEFAULT_CONFIG = SOURCE_ROOT / "learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml"


def build_training_command(config_path: str | Path, num_processes: int, exp_name: str, run_id: str, overrides: Sequence[str] = ()) -> list[str]:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if int(num_processes) < 1:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    invalid = [value for value in overrides if "=" not in value]
    if invalid:
        raise ValueError(f"training overrides must use key=value syntax: {invalid}")
    return [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(int(num_processes)), str(SOURCE_ROOT / "learning/training/trainer.py"), f"config={config_path}", f"exp_name={exp_name}", f"run_id={run_id}", *overrides]


def run_training(config_path: str | Path = DEFAULT_CONFIG, *, num_processes: int = 8, exp_name: str | None = None, run_id: str | None = None, no_wandb: bool = False, overrides: Sequence[str] = ()) -> None:
    launch_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if exp_name is None else str(exp_name)
    run_id = launch_id if run_id is None else str(run_id)
    command = build_training_command(config_path, num_processes, launch_id, run_id, (*overrides, *(("no_wandb=true",) if no_wandb else ())))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(SOURCE_ROOT), env.get("PYTHONPATH", "")))
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    subprocess.run(command, cwd=SOURCE_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--exp-name")
    parser.add_argument("--run-id")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run_training(args.config, num_processes=args.num_processes, exp_name=args.exp_name, run_id=args.run_id, no_wandb=args.no_wandb, overrides=args.override)


if __name__ == "__main__":
    main()
