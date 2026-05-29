# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_align_hand_depth(
    input_path: str,
    depth_path: str,
    intrinsics_path: str,
    output_path: str,
    mesh_intrinsics_path: str | None = None,
    per_frame: bool = False,
    per_hand: bool = False,
    align: str = 'scale',
    smooth_sigma: float = 0.0,
    diag_dir: str | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "input_path": input_path,
        "depth_path": depth_path,
        "intrinsics_path": intrinsics_path,
    }
    if mesh_intrinsics_path is not None:
        inputs["mesh_intrinsics_path"] = mesh_intrinsics_path
    extra: dict = {
        "per_frame": per_frame,
        "per_hand": per_hand,
        "align": align,
        "smooth_sigma": smooth_sigma if smooth_sigma > 0 else None,
    }
    outputs = {"output_path": output_path}
    if diag_dir is not None:
        outputs["diag_dir"] = diag_dir
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.align_hand_depth",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mesh_intrinsics_path", default=None)
    parser.add_argument("--per_frame", action="store_true")
    parser.add_argument("--per_hand", action="store_true")
    parser.add_argument("--align", default="scale", choices=["scale", "offset"])
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--diag_dir", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_hand_depth(
        args.input_path, args.depth_path, args.intrinsics_path, args.output_path,
        mesh_intrinsics_path=args.mesh_intrinsics_path,
        per_frame=args.per_frame,
        per_hand=args.per_hand,
        align=args.align,
        smooth_sigma=args.smooth_sigma,
        diag_dir=args.diag_dir,
        dev=args.dev,
    )
