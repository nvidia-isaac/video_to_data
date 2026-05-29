# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_videos_to_poses.yaml"

_DEV_PRESERVE_VOLUMES = [
    "/workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build",
    "/workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda",
]


def run_mv_videos_to_poses(
    camera_params_path: str,
    rgb_dir: str,
    depth_dir: str,
    mask_dir: str,
    mesh_path: str,
    weights_dir: str,
    output_dir: str,
    symmetry_path: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "rgb_dir": rgb_dir,
        "depth_dir": depth_dir,
        "mask_dir": mask_dir,
        "mesh_path": mesh_path,
        "symmetry_path": symmetry_path,
        "weights_dir": weights_dir,
        "config_path": config_path,
    }

    outputs = {"output_dir": output_dir}

    weights_abs = Path(weights_dir).resolve()
    weights_container = f"/data/weights_dir/{weights_abs.name}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.mv_videos_to_poses",
        inputs=inputs,
        outputs=outputs,
        extra_args={"debug": debug if debug >= 0 else None},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={
            "PYTHONUNBUFFERED": "1",
            "FOUNDATIONPOSE_WEIGHTS_DIR": weights_container,
        },
        extra_volumes=_DEV_PRESERVE_VOLUMES if dev else None,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-view FoundationPose tracking")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing input frames")
    parser.add_argument("--camera_params_path", type=str, required=True, help="Path to camera parameters")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing depth maps")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing object masks")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to object mesh file")
    parser.add_argument("--symmetry_path", type=str, default=None,
                        help="Optional BOP-style symmetry JSON; defaults to <mesh_dir>/output_symmetry.json if present")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing FoundationPose weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output poses")
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG), help="Path to config YAML")
    parser.add_argument("--debug", type=int, default=-1, help="Debug level")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_videos_to_poses(
        camera_params_path=args.camera_params_path,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir,
        mesh_path=args.mesh_path,
        symmetry_path=args.symmetry_path,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        debug=args.debug,
        dev=args.dev,
    )
