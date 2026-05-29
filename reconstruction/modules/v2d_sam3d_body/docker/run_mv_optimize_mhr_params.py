# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_optimize_mhr_params.yaml"


def run_mv_optimize_mhr_params(
    camera_params_path: str,
    rgb_dir: str,
    weights_dir: str,
    output_dir: str,
    bbox_dir: str | None = None,
    mask_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "rgb_dir": rgb_dir,
        "weights_dir": weights_dir,
        "config_path": config_path,
    }
    if bbox_dir:
        inputs["bbox_dir"] = bbox_dir
    if mask_dir:
        inputs["mask_dir"] = mask_dir

    outputs = {"output_dir": output_dir}

    weights_abs = Path(weights_dir).resolve()
    weights_container = f"/data/weights_dir/{weights_abs.name}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3d_body.lib.mv_optimize_mhr_params",
        inputs=inputs,
        outputs=outputs,
        extra_args={"debug": debug if debug >= 0 else None},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={
            "PYTHONUNBUFFERED": "1",
            "TORCH_HOME": f"{weights_container}/torch_home",
            "HF_HOME": f"{weights_container}/hf_home",
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-view MHR parameter optimization")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing input frames")
    parser.add_argument("--camera_params_path", type=str, required=True, help="Path to camera parameters")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--bbox_dir", type=str, default=None, help="Directory containing bounding boxes")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing SAM2 masks (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG), help="Path to config YAML")
    parser.add_argument("--debug", type=int, default=-1, help="Debug level")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_optimize_mhr_params(
        camera_params_path=args.camera_params_path,
        rgb_dir=args.rgb_dir,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        bbox_dir=args.bbox_dir,
        mask_dir=args.mask_dir,
        config_path=args.config_path,
        debug=args.debug,
        dev=args.dev,
    )
