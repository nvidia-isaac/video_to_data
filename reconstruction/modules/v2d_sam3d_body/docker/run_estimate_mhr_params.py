# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_estimate_mhr_params(
    rgb_path: str,
    weights_dir: str,
    output_params_path: str,
    cam_intrinsics_path: str | None = None,
    bbox_path: str | None = None,
    mask_path: str | None = None,
    output_mesh_path: str | None = None,
    batch_size: int = 1,
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
        "rgb_path": rgb_path,
        "weights_dir": weights_dir,
    }
    if cam_intrinsics_path is not None:
        inputs["cam_intrinsics_path"] = cam_intrinsics_path
    if bbox_path is not None:
        inputs["bbox_path"] = bbox_path
    if mask_path is not None:
        inputs["mask_path"] = mask_path

    outputs = {"output_params_path": output_params_path}
    if output_mesh_path:
        outputs["output_mesh_path"] = output_mesh_path

    extra_args = {"debug": debug if debug >= 0 else None}
    if batch_size != 1:
        extra_args["batch_size"] = batch_size

    weights_abs = Path(weights_dir).resolve()
    weights_container = f"/data/weights_dir/{weights_abs.name}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3d_body.lib.estimate_mhr_params",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra_args,
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

    parser = argparse.ArgumentParser(description="Run SAM3D-Body MHR estimation (single camera)")
    parser.add_argument("--rgb_path", type=str, required=True,
                        help="Path to input frames (image dir, .h5, or video file)")
    parser.add_argument("--cam_intrinsics_path", type=str, default=None,
                        help="Optional. If omitted, the model uses a default FOV.")
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--bbox_path", type=str, default=None,
                        help="Optional bbox track .pt path.")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Optional SAM2 mask directory or .h5 path.")
    parser.add_argument("--output_params_path", type=str, required=True)
    parser.add_argument("--output_mesh_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Frames per inference call (>1 amortizes Python dispatcher overhead).")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_estimate_mhr_params(
        rgb_path=args.rgb_path,
        cam_intrinsics_path=args.cam_intrinsics_path,
        weights_dir=args.weights_dir,
        bbox_path=args.bbox_path,
        mask_path=args.mask_path,
        output_params_path=args.output_params_path,
        output_mesh_path=args.output_mesh_path,
        batch_size=args.batch_size,
        debug=args.debug,
        dev=args.dev,
    )
