# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from v2d.docker.container import run_in_container
from v2d.hamer.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_hands_aligned_video(
    frames_dir: str,
    aligned_dir: str,
    mano_assets_root: str,
    output_path: str,
    object_mesh_path: str | None = None,
    object_poses_dir: str | None = None,
    object_scale: float = 1.0,
    fps: float = 30.0,
    alpha: float = 0.55,
    use_pre_dz_cam_t: bool = False,
    dev: bool = False,
) -> None:
    inputs = {
        "frames_dir":       frames_dir,
        "aligned_dir":      aligned_dir,
        "mano_assets_root": mano_assets_root,
    }
    if object_mesh_path is not None:
        inputs["object_mesh_path"] = object_mesh_path
    if object_poses_dir is not None:
        inputs["object_poses_dir"] = object_poses_dir
    extra_args: dict = {"fps": fps, "alpha": alpha, "object_scale": object_scale}
    if use_pre_dz_cam_t:
        extra_args["use_pre_dz_cam_t"] = True
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hamer.lib.render_hands_aligned_video",
        inputs=inputs,
        outputs={"output_path": output_path},
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render aligned MANO meshes onto video")
    parser.add_argument("--frames_dir",       required=True)
    parser.add_argument("--aligned_dir",      required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--object_mesh_path", default=None)
    parser.add_argument("--object_poses_dir", default=None)
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--fps",   type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--use_pre_dz_cam_t", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_hands_aligned_video(
        frames_dir       = args.frames_dir,
        aligned_dir      = args.aligned_dir,
        mano_assets_root = args.mano_assets_root,
        output_path      = args.output_path,
        object_mesh_path = args.object_mesh_path,
        object_poses_dir = args.object_poses_dir,
        object_scale     = args.object_scale,
        fps              = args.fps,
        alpha            = args.alpha,
        use_pre_dz_cam_t = args.use_pre_dz_cam_t,
        dev              = args.dev,
    )
