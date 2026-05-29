# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_export_soma(
    params_path: str,
    output_path: str,
    mesh_path: str | None = None,
    weights_dir: str | None = None,
    autograd_iters: int | None = None,
    leaf_weight: float | None = None,
    foot_weight: float | None = None,
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {"params_path": params_path}
    if mesh_path:
        inputs["mesh_path"] = mesh_path
    if weights_dir:
        inputs["weights_dir"] = weights_dir

    outputs = {"output_path": output_path}

    extra_args = {
        "autograd_iters": autograd_iters,
        "leaf_weight": leaf_weight,
        "foot_weight": foot_weight,
        "debug": debug if debug >= 0 else None,
    }

    env = {"PYTHONUNBUFFERED": "1"}
    if weights_dir:
        weights_abs = Path(weights_dir).resolve()
        weights_container = f"/data/weights_dir/{weights_abs.name}"
        env["TORCH_HOME"] = f"{weights_container}/torch_home"
        env["HF_HOME"] = f"{weights_container}/hf_home"

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3d_body.lib.export_soma",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env=env,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MHR to SOMA export in Docker")
    parser.add_argument("--params_path", type=str, required=True,
                        help="Path to mhr_params_mv.pt")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output .npz file path")
    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Path to mhr_mesh_mv.pt (optional)")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="sam3d_body weights directory (fallback for MHR JIT)")
    parser.add_argument("--autograd_iters", type=int, default=None,
                        help="Autograd FK refinement steps after analytical IK (default 0 = analytical only)")
    parser.add_argument("--leaf_weight", type=float, default=None,
                        help="Uniform extremity vertex weight passed to PoseInversion.fit")
    parser.add_argument("--foot_weight", type=float, default=None,
                        help="Override foot vertex weight; pair with --autograd_iters > 0")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_export_soma(
        params_path=args.params_path,
        output_path=args.output_path,
        mesh_path=args.mesh_path,
        weights_dir=args.weights_dir,
        autograd_iters=args.autograd_iters,
        leaf_weight=args.leaf_weight,
        foot_weight=args.foot_weight,
        debug=args.debug,
        dev=args.dev,
    )
