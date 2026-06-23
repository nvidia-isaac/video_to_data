# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docker wrapper for export_demo.

Load exported training data and render HOI overlay as a sanity check:
    python -m v2d.mv.postprocess.docker.run_export_demo \\
        --seq_dir /local/path/to/exported/sequence \\
        --output_dir /local/path/to/demo_output \\
        --dev

seq_dir is the flat directory produced by export_sequence (containing edex,
images/, poses.npy, mhr_mesh_mv.pt, etc.). Requires GPU for pyrender/EGL.
"""

from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR


def run_export_demo(
    seq_dir: str,
    output_dir: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.postprocess.lib.export_demo",
        inputs={"seq_dir": seq_dir},
        outputs={"output_dir": output_dir},
        gpus=True,
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run export_demo in Docker")
    parser.add_argument("--seq_dir", type=str, required=True,
                        help="Path to the exported sequence directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to write demo output (overlay videos)")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_export_demo(
        seq_dir=args.seq_dir,
        output_dir=args.output_dir,
        dev=args.dev,
    )
