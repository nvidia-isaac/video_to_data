# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host wrapper: run a hand-object dataset loader (Stage 1 / MANO FK) in-container.

Mounts the (separately-licensed, never-vendored) MANO model dir, the raw
human-motion-data dir, and the output dir, then runs the module-dispatch loader.
The container produces the ``${dataset}_loaded`` Parquet (ManoSharpaData with
MANO + object only) consumed downstream by robotic_grounding's IK retarget.

NOTE: the exact arg surface / data-dir layout is finalized when the OSMO load
workflow is wired (Phase 3).
"""
import os

from v2d.docker.container import run_in_container
from v2d.task_library_loader.docker._config import IMAGE_NAME, MODULES_DIR


def run_loader(
    dataset: str,
    output_dir: str,
    mano_model_dir: str,
    human_motion_data_dir: str,
    device: str = "cuda:0",
    save: bool = True,
    sequence_pattern: str | None = None,
    sequence_id: str | None = None,
    max_sequences: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.task_library_loader.lib.run_loader",
        inputs={"mano_model_dir": mano_model_dir},
        outputs={"output_dir": output_dir},
        extra_args={
            "dataset": dataset,
            "device": device,
            "save": save,
            "sequence_pattern": sequence_pattern,
            "sequence_id": sequence_id,
            "max_sequences": max_sequences,
            "shard_id": shard_id,
            "num_shards": num_shards,
        },
        # The loaders read raw inputs from $HUMAN_MOTION_DATA_DIR/<dataset>/...
        env={"HUMAN_MOTION_DATA_DIR": "/data/human_motion_data"},
        extra_volumes=[
            f"{os.path.abspath(human_motion_data_dir)}:/data/human_motion_data"
        ],
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mano_model_dir", required=True)
    parser.add_argument("--human_motion_data_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--sequence_pattern", default=None)
    parser.add_argument("--sequence_id", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_loader(
        dataset=args.dataset,
        output_dir=args.output_dir,
        mano_model_dir=args.mano_model_dir,
        human_motion_data_dir=args.human_motion_data_dir,
        device=args.device,
        save=args.save,
        sequence_pattern=args.sequence_pattern,
        sequence_id=args.sequence_id,
        max_sequences=args.max_sequences,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        dev=args.dev,
    )
