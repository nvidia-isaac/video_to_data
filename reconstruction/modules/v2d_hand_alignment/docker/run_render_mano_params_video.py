# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_mano_params_video(
    mano_params_path: str,
    intrinsics_path: str,
    mano_model_dir: str,
    output_path: str,
    fps: float = 25.0,
    frames_folder: str | None = None,
    mesh_path: str | None = None,
    poses_dir: str | None = None,
    panel_w: int = 388,
    panel_h: int = 516,
    start: int = 0,
    end: int | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "mano_params": mano_params_path,
        "intrinsics": intrinsics_path,
        "mano_model_dir": mano_model_dir,
    }
    if frames_folder is not None:
        inputs["frames_folder"] = frames_folder
    if mesh_path is not None:
        inputs["mesh"] = mesh_path
    if poses_dir is not None:
        inputs["poses_dir"] = poses_dir
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.render_mano_params_video",
        inputs=inputs,
        outputs={"output": output_path},
        extra_args={
            "fps": fps,
            "panel_w": panel_w,
            "panel_h": panel_h,
            "start": start,
            "end": end,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
        env={"PYOPENGL_PLATFORM": "egl"},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mano_params_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--mano_model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frames_folder", default=None)
    parser.add_argument("--mesh_path", default=None)
    parser.add_argument("--poses_dir", default=None)
    parser.add_argument("--panel_w", type=int, default=388)
    parser.add_argument("--panel_h", type=int, default=516)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_mano_params_video(
        mano_params_path=args.mano_params_path,
        intrinsics_path=args.intrinsics_path,
        mano_model_dir=args.mano_model_dir,
        output_path=args.output_path,
        fps=args.fps,
        frames_folder=args.frames_folder,
        mesh_path=args.mesh_path,
        poses_dir=args.poses_dir,
        panel_w=args.panel_w,
        panel_h=args.panel_h,
        start=args.start,
        end=args.end,
        dev=args.dev,
    )
