# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host wrapper: run a hand-object dataset loader (Stage 1 / MANO FK) in-container.

Mounts the (separately-licensed, never-vendored) MANO model dir, the raw
human-motion-data dir, and the output dir, then runs the module-dispatch loader.
The container produces the ``${dataset}_loaded`` Parquet (ManoSharpaData with
MANO + object only) consumed downstream by robotic_grounding's IK retarget.
"""
import os

from v2d.docker.container import run_in_container
from v2d.task_library_loader.docker._config import IMAGE_NAME, MODULES_DIR


def run_loader(
    dataset: str,
    output_dir: str,
    mano_model_dir: str,
    human_motion_data_dir: str,
    object_assets_dir: str | None = None,
    device: str = "cuda:0",
    save: bool = True,
    sequence_pattern: str | None = None,
    sequence_id: str | None = None,
    max_sequences: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    dev: bool = False,
) -> None:
    extra_volumes = [
        f"{os.path.abspath(human_motion_data_dir)}:/data/human_motion_data"
    ]
    extra_args: dict[str, object] = {
        "dataset": dataset,
        "device": device,
        "save": save,
        "sequence_pattern": sequence_pattern,
        "sequence_id": sequence_id,
        "max_sequences": max_sequences,
        "shard_id": shard_id,
        "num_shards": num_shards,
    }

    # Object assets (rigid URDFs + meshes). Mount the root containing
    # `urdfs/<dataset>/` and `meshes/<dataset>/` as ONE volume so the URDFs'
    # relative `../../meshes/<dataset>/…` refs resolve (MuJoCo loads arctic's
    # articulated URDF by path and follows those refs).
    if object_assets_dir is not None:
        extra_volumes.append(
            f"{os.path.abspath(object_assets_dir)}:/data/object_assets"
        )
        extra_args["object_model_root"] = f"/data/object_assets/urdfs/{dataset}"
        extra_args["mesh_dir"] = f"/data/object_assets/meshes/{dataset}"

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.task_library_loader.lib.run_loader",
        inputs={"mano_model_dir": mano_model_dir},
        outputs={"output_dir": output_dir},
        extra_args=extra_args,
        # The loaders read raw inputs from $HUMAN_MOTION_DATA_DIR/<dataset>/...
        env={"HUMAN_MOTION_DATA_DIR": "/data/human_motion_data"},
        extra_volumes=extra_volumes,
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
    parser.add_argument(
        "--object_assets_dir",
        default=None,
        help="Root holding urdfs/<dataset>/ + meshes/<dataset>/ (mounted as one "
        "volume so URDF '../../meshes/...' refs resolve). Omit for h2o/grab/dexycb.",
    )
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
        object_assets_dir=args.object_assets_dir,
        device=args.device,
        save=args.save,
        sequence_pattern=args.sequence_pattern,
        sequence_id=args.sequence_id,
        max_sequences=args.max_sequences,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        dev=args.dev,
    )
