# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.mesh.docker._config import IMAGE_NAME, MODULES_DIR


def run_mesh_simplify(
    input_mesh_path: str,
    output_mesh_path: str,
    face_count: int | None = None,
    factor: float | None = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mesh.lib.run_mesh_simplify",
        inputs={"input_mesh": input_mesh_path},
        outputs={"output_mesh": output_mesh_path},
        extra_args={"face_count": face_count, "factor": factor},
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplify a mesh (via Docker)")
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_mesh", required=True)
    parser.add_argument("--face_count", type=int, default=None)
    parser.add_argument("--factor", type=float, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_simplify(args.input_mesh, args.output_mesh, face_count=args.face_count, factor=args.factor, dev=args.dev)
