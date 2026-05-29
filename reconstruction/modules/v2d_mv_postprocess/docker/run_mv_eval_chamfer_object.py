# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_eval_chamfer_object.yaml"


def run_mv_eval_chamfer_object(
    camera_params_path: str,
    object_mesh_path: str,
    object_pose_dir: str,
    output_dir: str,
    depth_dir: str,
    mask_dir: str,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "object_mesh_path": object_mesh_path,
        "object_pose_dir": object_pose_dir,
        "config_path": config_path,
        "depth_dir": depth_dir,
        "mask_dir": mask_dir,
    }
    outputs = {"output_dir": output_dir}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.postprocess.lib.mv_eval_chamfer_object",
        inputs=inputs,
        outputs=outputs,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chamfer distance evaluation for object mesh"
    )
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str, required=True)
    parser.add_argument("--object_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_eval_chamfer_object(
        camera_params_path=args.camera_params_path,
        object_mesh_path=args.object_mesh_path,
        object_pose_dir=args.object_pose_dir,
        output_dir=args.output_dir,
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
