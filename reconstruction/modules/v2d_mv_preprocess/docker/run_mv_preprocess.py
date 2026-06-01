# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.preprocess.docker._config import IMAGE_NAME, MODULES_DIR

_DEFAULT_CONFIG = Path(__file__).parent.parent / "lib" / "mv_preprocess.yaml"


def run_mv_preprocess(
    rgb_dir: str,
    output_dir: str,
    config_path: str = str(_DEFAULT_CONFIG),
    camera_params_path: str | None = None,
    extrinsics_camera_params_path: str | None = None,
    hoi_metadata_path: str | None = None,
    mesh_path: str | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "rgb_dir": rgb_dir,
        "config_path": config_path,
    }
    if camera_params_path is not None:
        inputs["camera_params_path"] = camera_params_path
    if extrinsics_camera_params_path is not None:
        inputs["extrinsics_camera_params_path"] = extrinsics_camera_params_path
    if hoi_metadata_path is not None:
        inputs["hoi_metadata_path"] = hoi_metadata_path
    if mesh_path is not None:
        inputs["mesh_path"] = mesh_path

    outputs = {
        "output_dir": output_dir,
    }

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.preprocess.lib.mv_preprocess",
        inputs=inputs,
        outputs=outputs,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-view preprocessing in Docker")
    parser.add_argument("--rgb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--camera_params_path", type=str, default=None)
    parser.add_argument("--extrinsics_camera_params_path", type=str, default=None)
    parser.add_argument("--hoi_metadata_path", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_preprocess(
        rgb_dir=args.rgb_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        camera_params_path=args.camera_params_path,
        extrinsics_camera_params_path=args.extrinsics_camera_params_path,
        hoi_metadata_path=args.hoi_metadata_path,
        mesh_path=args.mesh_path,
        dev=args.dev,
    )
