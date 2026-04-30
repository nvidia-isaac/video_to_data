"""Docker wrapper for export_sequence.

Export from CSS (remote):
    python -m v2d.mv.postprocess.docker.run_export_sequence \\
        --swift_output_base swift://pdx.s8k.io/AUTH_.../data_output/<seq> \\
        --output_dir /local/path/to/sequence \\
        --dev

Export from local directory:
    python -m v2d.mv.postprocess.docker.run_export_sequence \\
        --source_dir /path/to/osmo/task/outputs \\
        --output_dir /local/path/to/sequence \\
        --dev

Requires CSS_ACCESS_KEY and CSS_SECRET_KEY env vars for remote mode.
"""

import os
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR


def run_export_sequence(
    output_dir: str,
    swift_output_base: str | None = None,
    source_dir: str | None = None,
    dry_run: bool = False,
    max_workers: int | None = None,
    final_only: bool = False,
    dev: bool = False,
) -> None:
    inputs = {}
    extra_args = {}
    env = {}

    if swift_output_base is not None:
        extra_args["swift_output_base"] = swift_output_base
        env = {
            "CSS_ACCESS_KEY": os.environ.get("CSS_ACCESS_KEY", ""),
            "CSS_SECRET_KEY": os.environ.get("CSS_SECRET_KEY", ""),
            "CSS_ENDPOINT_URL": os.environ.get("CSS_ENDPOINT_URL", "https://pdx.s8k.io"),
        }
    elif source_dir is not None:
        inputs["source_dir"] = source_dir

    if dry_run:
        extra_args["dry_run"] = True
    if max_workers is not None:
        extra_args["max_workers"] = max_workers
    if final_only:
        extra_args["final_only"] = True

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.postprocess.lib.export_sequence",
        inputs=inputs,
        outputs={"output_dir": output_dir},
        extra_args=extra_args,
        env=env or None,
        gpus=False,
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run export_sequence in Docker"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--swift_output_base", type=str,
                        help="Swift URL for remote download")
    source.add_argument("--source_dir", type=str,
                        help="Local directory containing OSMO task outputs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local output directory")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel download threads (default: CPU count)")
    parser.add_argument("--final_only", action="store_true",
                        help="Export only final outputs (trajectories, ground plane, object mesh, "
                             "edex, tiled overlay); skip depth/masks/images/videos.")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_export_sequence(
        output_dir=args.output_dir,
        swift_output_base=args.swift_output_base,
        source_dir=args.source_dir,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
        final_only=args.final_only,
        dev=args.dev,
    )
