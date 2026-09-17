# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_export_soma(
    params_path: str,
    output_path: str,
    mesh_path: str | None = None,
    weights_dir: str | None = None,
    body_iters: int | None = None,
    finger_iters: int | None = None,
    full_iters: int | None = None,
    lie_iters: int | None = None,
    lie_lambda: float | None = None,
    autograd_iters: int | None = None,
    autograd_lr: float | None = None,
    autograd_translation_lr_scale: float | None = None,
    autograd_pose_prior: float | None = None,
    autograd_pose_prior_weights: str | None = None,
    autograd_hand_weight: float | None = None,
    autograd_foot_weight: float | None = None,
    leaf_weight: float | None = None,
    foot_weight: float | None = None,
    device: str | None = None,
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
        "body_iters": body_iters,
        "finger_iters": finger_iters,
        "full_iters": full_iters,
        "lie_iters": lie_iters,
        "lie_lambda": lie_lambda,
        "autograd_iters": autograd_iters,
        "autograd_lr": autograd_lr,
        "autograd_translation_lr_scale": autograd_translation_lr_scale,
        "autograd_pose_prior": autograd_pose_prior,
        "autograd_pose_prior_weights": autograd_pose_prior_weights,
        "autograd_hand_weight": autograd_hand_weight,
        "autograd_foot_weight": autograd_foot_weight,
        "leaf_weight": leaf_weight,
        "foot_weight": foot_weight,
        "device": device,
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
    parser.add_argument("--body-iters", "--body_iters", dest="body_iters",
                        type=int, default=None)
    parser.add_argument("--finger-iters", "--finger_iters", dest="finger_iters",
                        type=int, default=None)
    parser.add_argument("--full-iters", "--full_iters", dest="full_iters",
                        type=int, default=None)
    parser.add_argument("--lie-iters", "--lie_iters", dest="lie_iters",
                        type=int, default=None)
    parser.add_argument("--lie-lambda", "--lie_lambda", dest="lie_lambda",
                        type=float, default=None)
    parser.add_argument("--autograd-iters", "--autograd_iters",
                        dest="autograd_iters", type=int, default=None,
                        help="Autograd FK refinement steps after analytical IK (default 0 = analytical only)")
    parser.add_argument("--autograd-lr", "--autograd_lr",
                        dest="autograd_lr", type=float, default=None)
    parser.add_argument(
        "--autograd-translation-lr-scale", "--autograd_translation_lr_scale",
        dest="autograd_translation_lr_scale", type=float, default=None,
    )
    parser.add_argument(
        "--autograd-pose-prior", "--autograd_pose_prior",
        dest="autograd_pose_prior", type=float, default=None,
    )
    parser.add_argument(
        "--autograd-pose-prior-weights", "--autograd_pose_prior_weights",
        dest="autograd_pose_prior_weights", default=None,
    )
    parser.add_argument(
        "--autograd-hand-weight", "--autograd_hand_weight",
        dest="autograd_hand_weight", type=float, default=None,
    )
    parser.add_argument(
        "--autograd-foot-weight", "--autograd_foot_weight",
        dest="autograd_foot_weight", type=float, default=None,
    )
    parser.add_argument("--leaf_weight", type=float, default=None,
                        help="Uniform extremity vertex weight passed to PoseInversion.fit")
    parser.add_argument("--foot_weight", type=float, default=None,
                        help="Override foot vertex weight; pair with --autograd_iters > 0")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_export_soma(
        params_path=args.params_path,
        output_path=args.output_path,
        mesh_path=args.mesh_path,
        weights_dir=args.weights_dir,
        body_iters=args.body_iters,
        finger_iters=args.finger_iters,
        full_iters=args.full_iters,
        lie_iters=args.lie_iters,
        lie_lambda=args.lie_lambda,
        autograd_iters=args.autograd_iters,
        autograd_lr=args.autograd_lr,
        autograd_translation_lr_scale=args.autograd_translation_lr_scale,
        autograd_pose_prior=args.autograd_pose_prior,
        autograd_pose_prior_weights=args.autograd_pose_prior_weights,
        autograd_hand_weight=args.autograd_hand_weight,
        autograd_foot_weight=args.autograd_foot_weight,
        leaf_weight=args.leaf_weight,
        foot_weight=args.foot_weight,
        device=args.device,
        debug=args.debug,
        dev=args.dev,
    )
